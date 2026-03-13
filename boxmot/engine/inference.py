# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

"""
Unified inference module for BoxMOT.

Provides a consistent interface for running detection across both real-time
tracking (tracker.py) and batch evaluation (evaluator.py) workflows.
No dependency on Ultralytics internals — all framework-specific code lives
inside the individual detector classes.
"""

import time
import types
from pathlib import Path
from typing import Callable, Generator, List, Optional, Union

import cv2
import numpy as np
import torch

from boxmot.detectors import default_imgsz, get_detector_class
from boxmot.detectors.detector import Detections
from boxmot.utils import logger as LOGGER
from boxmot.utils.timing import TimingStats
from boxmot.utils.torch_utils import select_device


# ---------------------------------------------------------------------------
# ReID timing wrapper
# ---------------------------------------------------------------------------

def resolve_yolo_model_path(yolo_model_path: Union[str, Path]) -> Path:
    """Resolve detector weights while preserving RT-DETR model names."""
    model_path = Path(yolo_model_path)

    if is_rtdetr_model(yolo_model_path):
        return Path(model_path.name)

    return WEIGHTS / model_path.name


class TimedReIDModel:
    """Wraps a ReID model to instrument get_features() timing."""

    def __init__(self, model, timing_stats: Optional[TimingStats] = None):
        self._model = model
        self._timing_stats = timing_stats

    def get_features(self, xyxys: np.ndarray, img: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        features = self._model.get_features(xyxys, img)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if self._timing_stats is not None:
            self._timing_stats.add_reid_time(elapsed_ms)
        return features

    def __getattr__(self, name):
        return getattr(self._model, name)


# ---------------------------------------------------------------------------
# Predictor proxy — passed to callbacks instead of an Ultralytics predictor
# ---------------------------------------------------------------------------

class _PredictorProxy:
    """
    Lightweight object passed to on_predict_start / on_predict_postprocess_end
    callbacks in place of the Ultralytics predictor.

    Carries only the state that BoxMOT callbacks actually need.
    """

    def __init__(self, device: str, bs: int = 1):
        self.device = device
        self.dataset = types.SimpleNamespace(bs=bs)
        self.results: list = []
        self.trackers: list = []
        self.custom_args = None


# ---------------------------------------------------------------------------
# Source iterator — yields (path, frame) from any common source type
# ---------------------------------------------------------------------------

def _iter_source(source, vid_stride: int = 1):
    """
    Yield (path, numpy_bgr_frame) pairs for every selected frame in *source*.

    Supported source types:
      - np.ndarray                single image
      - list[np.ndarray]          batch of images (each yielded separately)
      - str/Path  →  image file   single image
      - str/Path  →  directory    all images inside (sorted)
      - str/Path  →  video file   all frames (respecting vid_stride)
      - int                       webcam index
    """
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    if isinstance(source, np.ndarray):
        yield "", source
        return

    if isinstance(source, list):
        for item in source:
            if isinstance(item, np.ndarray):
                yield "", item
            else:
                img = cv2.imread(str(item))
                if img is not None:
                    yield str(item), img
        return

    # Integer → webcam
    try:
        cam_id = int(source)
        cap = cv2.VideoCapture(cam_id)
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if (frame_idx - 1) % vid_stride == 0:
                yield str(cam_id), frame
        cap.release()
        return
    except (TypeError, ValueError):
        pass

    p = Path(str(source))

    # Directory of images
    if p.is_dir():
        paths = sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXTS)
        for fp in paths:
            img = cv2.imread(str(fp))
            if img is not None:
                yield str(fp), img
        return

    # Single image file
    img = cv2.imread(str(p))
    if img is not None:
        yield str(p), img
        return

    # Video file
    cap = cv2.VideoCapture(str(p))
    if cap.isOpened():
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if (frame_idx - 1) % vid_stride == 0:
                yield str(p), frame
        cap.release()
    else:
        LOGGER.error(f"Could not open source: {source}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class DetectorReIDPipeline:
    """
    Unified pipeline for detection and optional ReID inference.

    Works with any BoxMOT detector (YOLOX, RT-DETR, Ultralytics YOLO) through
    a single Detector interface.
    """

    def __init__(
        self,
        detector_path: Union[str, Path],
        reid_paths: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
        device: str = "",
        imgsz: Optional[Union[int, List[int]]] = None,
        half: bool = False,
        timing_stats: Optional[TimingStats] = None,
    ):
        self.detector_path = Path(detector_path)
        self.device = select_device(device)
        self.half = half
        self.timing_stats = timing_stats if timing_stats is not None else TimingStats()

        if imgsz is None:
            imgsz = default_imgsz(detector_path)
        self.imgsz = imgsz

        # Instantiate the appropriate detector
        detector_class = get_detector_class(self.detector_path)
        self.detector = detector_class(
            model=self.detector_path,
            device=self.device,
            imgsz=self.imgsz,
        )

        # Callback registry
        self._callbacks: dict[str, list[Callable]] = {
            "on_predict_start": [],
            "on_predict_postprocess_end": [],
        }

        # ReID models
        self.reid_models: List[TimedReIDModel] = []
        self.reid_model_names: List[str] = []
        if reid_paths is not None:
            self._init_reid_models(reid_paths)

    # ------------------------------------------------------------------
    # ReID initialization
    # ------------------------------------------------------------------

    def _init_reid_models(self, reid_model_paths):
        from boxmot.reid.core.auto_backend import ReidAutoBackend

        if isinstance(reid_model_paths, (str, Path)):
            reid_model_paths = [reid_model_paths]

        for reid_path in reid_model_paths:
            reid_path = Path(reid_path)
            backend = ReidAutoBackend(
                weights=reid_path, device=self.device, half=self.half
            )
            self.reid_models.append(TimedReIDModel(backend.model, self.timing_stats))
            self.reid_model_names.append(reid_path.stem)

    # ------------------------------------------------------------------
    # Callback management
    # ------------------------------------------------------------------

    def add_callback(self, event: str, callback: Callable):
        if event in self._callbacks:
            self._callbacks[event].append(callback)
        else:
            LOGGER.warning(f"Unknown callback event '{event}'. Ignoring.")

    def _fire(self, event: str, proxy: _PredictorProxy):
        for cb in self._callbacks[event]:
            cb(proxy)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        source,
        conf: float,
        iou: float,
        agnostic_nms: bool,
        classes: Optional[List[int]],
        vid_stride: int = 1,
    ) -> Generator:
        """Stream detection results over *source* frame by frame."""
        proxy = _PredictorProxy(device=self.device, bs=1)
        self._fire("on_predict_start", proxy)

        for path, frame in _iter_source(source, vid_stride=vid_stride):
            self.timing_stats.start_frame()
            results = self.detector(
                [frame], conf=conf, iou=iou,
                classes=classes, agnostic_nms=agnostic_nms,
            )
            proxy.results = results
            self._fire("on_predict_postprocess_end", proxy)

            if getattr(getattr(proxy, "custom_args", None), "_user_quit", False):
                break

            yield from results

    def predict_batch(
        self,
        images: List[np.ndarray],
        conf: float,
        iou: float,
        agnostic_nms: bool,
        classes: Optional[List[int]],
    ) -> list:
        """Run batch detection and return a list of Detection objects."""
        return self.detector(
            images, conf=conf, iou=iou,
            classes=classes, agnostic_nms=agnostic_nms,
        )

    # ------------------------------------------------------------------
    # Warmup & batch size tuning
    # ------------------------------------------------------------------

    def warmup(self):
        """Warmup the detector with a dummy image."""
        if isinstance(self.imgsz, (list, tuple)):
            h, w = int(self.imgsz[0]), int(self.imgsz[1])
        else:
            h = w = int(self.imgsz)
        dummy = [np.zeros((h, w, 3), dtype=np.uint8)]
        try:
            self.detector(dummy, conf=0.25, iou=0.7, classes=None, agnostic_nms=False)
        except Exception as e:
            LOGGER.warning(f"Warmup failed: {e}")

    def autotune_batch_size(self, requested_batch_size: int) -> int:
        """Find the largest batch size that fits in memory."""
        dev_lower = str(self.device).lower()
        use_accel = dev_lower.startswith(
            ("cuda", "0", "1", "2", "3", "4", "5", "6", "7", "mps", "metal")
        )
        if not use_accel:
            return max(1, requested_batch_size)

        def _empty_cache():
            if dev_lower.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif dev_lower.startswith(("mps", "metal")) and hasattr(torch, "mps"):
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass

        if isinstance(self.imgsz, (list, tuple)):
            h, w = int(self.imgsz[0]), int(self.imgsz[1])
        else:
            h = w = int(self.imgsz)
        dummy = np.zeros((h, w, 3), dtype=np.uint8)

        bs = max(1, int(requested_batch_size))
        while bs >= 1:
            try:
                self.detector([dummy] * bs, conf=0.25, iou=0.7, classes=None, agnostic_nms=False)
                if bs < requested_batch_size:
                    LOGGER.info(f"Auto-tuned batch size: {bs} (requested: {requested_batch_size})")
                return bs
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                _empty_cache()
                next_bs = max(1, bs - 1)
                LOGGER.warning(f"Batch size {bs} OOM; trying {next_bs}.")
                if next_bs == bs:
                    break
                bs = next_bs

        raise RuntimeError("Unable to run even batch size 1; reduce image size or move to CPU.")

    # ------------------------------------------------------------------
    # ReID helpers
    # ------------------------------------------------------------------

    def get_reid_features(
        self,
        xyxys: np.ndarray,
        img: np.ndarray,
        reid_index: int = 0,
    ) -> np.ndarray:
        if not self.reid_models:
            raise ValueError("No ReID models initialized.")
        if reid_index >= len(self.reid_models):
            raise ValueError(f"ReID index {reid_index} out of range.")
        return self.reid_models[reid_index].get_features(xyxys, img)

    def get_all_reid_features(self, xyxys: np.ndarray, img: np.ndarray) -> dict:
        return {
            name: model.get_features(xyxys, img)
            for model, name in zip(self.reid_models, self.reid_model_names)
        }

    def print_timing_summary(self):
        self.timing_stats.print_summary()


# ---------------------------------------------------------------------------
# Utility functions (used by evaluator.py and tracker.py)
# ---------------------------------------------------------------------------

def prepare_detections(result: Detections, img: np.ndarray) -> np.ndarray:
    """
    Extract detections from a result and sanitize them for downstream use.

    For AABB (N, 6) — [x1, y1, x2, y2, conf, cls]:
      removes boxes where x2 <= x1, y2 <= y1, or area < 10 px².

    For OBB (N, 7) — [cx, cy, w, h, angle, conf, cls]:
      removes boxes where w <= 0, h <= 0, or w*h < 10 px².

    Returns filtered array of the same width, or empty (0, 6)/(0, 7) when
    no valid detections remain.
    """
    dets = result.dets
    if dets is None or len(dets) == 0:
        n_cols = dets.shape[1] if dets is not None and dets.ndim == 2 else 6
        return np.empty((0, n_cols))

    if result.is_obb:
        w, h = dets[:, 2], dets[:, 3]
        valid = (w > 0) & (h > 0) & (w * h >= 10.0)
    else:
        x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
        valid = (x2 > x1) & (y2 > y1) & ((x2 - x1) * (y2 - y1) >= 10.0)

    return dets[valid]
