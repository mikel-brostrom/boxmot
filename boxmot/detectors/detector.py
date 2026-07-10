from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from boxmot.data import IMAGE_EXTS, iter_source
from boxmot.detectors.base import BaseDetectorBackend, Detections, load_weights, resolve_image
from boxmot.detectors.registry import default_conf, default_imgsz, get_detector_class
from boxmot.utils import logger as LOGGER


def _is_single_inference_source(source: Any) -> bool:
    """Return whether ``source`` should keep the single-result return contract."""
    if isinstance(source, np.ndarray):
        return source.ndim == 3

    if isinstance(source, (str, Path)):
        source_str = str(source)
        if "://" in source_str or any(ch in source_str for ch in "*?[]"):
            return False

        path = Path(source_str)
        return path.is_file() and path.suffix.lower() in IMAGE_EXTS

    return False


def _iter_batches(source: Any, batch_size: int, vid_stride: int) -> Iterator[tuple[list[str], list[np.ndarray]]]:
    """Yield ``(paths, frames)`` batches from a detector source."""
    paths: list[str] = []
    frames: list[np.ndarray] = []

    for path, frame in iter_source(source, vid_stride=vid_stride):
        paths.append(path)
        frames.append(frame)
        if len(frames) >= batch_size:
            yield paths, frames
            paths, frames = [], []

    if frames:
        yield paths, frames


class Detector:
    """Public detector wrapper with overrideable stage hooks and source streaming."""

    def __init__(
        self,
        path: str | Path,
        device: str = "cpu",
        imgsz=None,
        conf: Optional[float] = None,
        iou: float = 0.7,
        classes=None,
        agnostic_nms: bool = False,
        batch: int = 1,
        vid_stride: int = 1,
        callbacks: Optional[dict[str, list[Callable[["Detector"], None]]]] = None,
    ) -> None:
        self.path = Path(path)
        self.device = device
        self.imgsz = default_imgsz(path) if imgsz is None else imgsz
        self.conf = default_conf(path) if conf is None else float(conf)
        self.iou = float(iou)
        self.classes = classes
        self.agnostic_nms = bool(agnostic_nms)
        self.batch_size = max(int(batch), 1)
        self.vid_stride = max(int(vid_stride), 1)
        self.backend = self._get_backend_class(path)(model=path, device=device, imgsz=self.imgsz)
        self.model = getattr(self.backend, "model", getattr(self.backend, "_yolo", self.backend))
        self.done_warmup = False
        self.dataset = None
        self.results = None
        self.raw_results = None
        self.batch = None
        self.seen = 0
        self.stream = False
        self.callbacks = callbacks or {
            "on_predict_start": [],
            "on_predict_batch_start": [],
            "on_predict_postprocess_end": [],
            "on_predict_end": [],
        }
        self._lock = threading.Lock()
        self._last_orig_imgs: list[np.ndarray] | None = None

    @classmethod
    def _get_backend_class(cls, path: str | Path):
        return get_detector_class(path)

    @staticmethod
    def _as_result_list(results):
        return results if isinstance(results, list) else [results]

    @staticmethod
    def _batch_input(frames: list[np.ndarray]):
        return frames[0] if len(frames) == 1 else frames

    def setup_source(self, source, batch: Optional[int] = None, vid_stride: Optional[int] = None):
        """Prepare a batched source iterator for predictor-style inference."""
        self.dataset = _iter_batches(
            source,
            batch_size=max(int(self.batch_size if batch is None else batch), 1),
            vid_stride=max(int(self.vid_stride if vid_stride is None else vid_stride), 1),
        )
        return self.dataset

    def run_callbacks(self, event: str) -> None:
        """Run registered callbacks for a predictor lifecycle event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def add_callback(self, event: str, func: Callable[["Detector"], None]) -> None:
        """Register a callback for a predictor lifecycle event."""
        self.callbacks.setdefault(event, []).append(func)

    def warmup(self) -> None:
        """Warm up the detector backend with a dummy frame once."""
        if self.done_warmup:
            return

        if isinstance(self.imgsz, (list, tuple)):
            height, width = int(self.imgsz[0]), int(self.imgsz[1])
        else:
            height = width = int(self.imgsz)

        dummy = np.zeros((height, width, 3), dtype=np.uint8)
        try:
            self.backend(
                [dummy],
                conf=self.conf,
                iou=self.iou,
                classes=self.classes,
                agnostic_nms=self.agnostic_nms,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(f"Detector warmup failed: {exc}")
        finally:
            self.done_warmup = True

    def preprocess(self, image: np.ndarray, **kwargs):
        images = image if isinstance(image, list) else [image]
        self._last_orig_imgs = images
        backend_pre = getattr(self.backend, "preprocess", None)
        if not callable(backend_pre):
            return image
        try:
            return backend_pre(images)
        except (TypeError, NotImplementedError):
            # Backend without a real preprocess stage (legacy contract):
            # fall back to the no-op pass-through so the composite path
            # ``self.backend(...)`` continues to work in ``process``.
            return image

    def process(self, frame, **kwargs):
        backend_proc = getattr(self.backend, "process", None)
        # Composite path: callers passing inference overrides
        # (conf/iou/classes/agnostic_nms) get the legacy "do everything"
        # semantics, which is what the warmup and standalone
        # ``Detector.process(images, conf=..., ...)`` callers rely on.
        composite_keys = {"conf", "iou", "classes", "agnostic_nms"}
        if any(key in kwargs for key in composite_keys):
            images = frame if isinstance(frame, list) else [frame]
            results = self.backend(
                images,
                conf=float(kwargs.get("conf", self.conf)),
                iou=float(kwargs.get("iou", self.iou)),
                classes=kwargs.get("classes", self.classes),
                agnostic_nms=bool(kwargs.get("agnostic_nms", self.agnostic_nms)),
            )
            if isinstance(results, list) and len(results) == 1:
                return results[0]
            return results
        # Stage path: ``frame`` is the output of ``self.preprocess`` and we
        # want only the model forward so timing reports inference separately
        # from preprocess/postprocess.
        if callable(backend_proc):
            try:
                return backend_proc(frame)
            except (TypeError, NotImplementedError):
                pass
        # Backend has no standalone process stage: fall back to composite.
        images = self._last_orig_imgs or (frame if isinstance(frame, list) else [frame])
        results = self.backend(
            images,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            agnostic_nms=self.agnostic_nms,
        )
        if isinstance(results, list) and len(results) == 1:
            return results[0]
        return results

    def postprocess(self, results, as_detections: bool = False, **kwargs):
        if results is None:
            dets = np.empty((0, 6), dtype=np.float32)
            if as_detections:
                orig_img = kwargs.get("image")
                if orig_img is None:
                    frames = kwargs.get("frames")
                    if isinstance(frames, (list, tuple)) and len(frames) > 0:
                        orig_img = frames[0]
                    else:
                        orig_img = np.empty((0, 0, 3), dtype=np.uint8)
                return Detections(dets=dets, orig_img=orig_img, path=str(kwargs.get("path", "")))
            return dets

        backend_post = getattr(self.backend, "postprocess", None)
        # If the backend has a real postprocess stage, route the raw model
        # output through it so NMS/scale-back work shows up in the dedicated
        # timing bucket. Backends without a real stage fall through to the
        # legacy unwrap-only behaviour.
        if callable(backend_post) and not isinstance(results, (Detections,)):
            already_detections = (
                isinstance(results, list)
                and len(results) > 0
                and all(isinstance(r, Detections) for r in results)
            )
            if not already_detections:
                try:
                    results = backend_post(
                        results,
                        conf=float(kwargs.get("conf", self.conf)),
                        iou=float(kwargs.get("iou", self.iou)),
                        classes=kwargs.get("classes", self.classes),
                        agnostic_nms=bool(kwargs.get("agnostic_nms", self.agnostic_nms)),
                    )
                except (TypeError, NotImplementedError):
                    pass
        if as_detections:
            if isinstance(results, list) and len(results) == 1:
                return results[0]
            return results
        if isinstance(results, Detections):
            if results.masks is not None:
                return results
            return results.dets
        if hasattr(results, "dets"):
            if getattr(results, "masks", None) is not None:
                return results
            return results.dets
        if isinstance(results, list) and all(isinstance(result, Detections) for result in results):
            if len(results) == 1:
                if results[0].masks is not None:
                    return results[0]
                return results[0].dets
            return results
        if isinstance(results, list) and all(hasattr(result, "dets") for result in results):
            if len(results) == 1:
                if getattr(results[0], "masks", None) is not None:
                    return results[0]
                return results[0].dets
            return [result.dets for result in results]
        return results

    def _predict_single(self, source, **kwargs):
        path = str(source) if isinstance(source, (str, Path)) else ""
        image = resolve_image(source)

        with self._lock:
            self.stream = False
            self.batch = ([path], [image])
            self.seen = 0
            self.run_callbacks("on_predict_start")
            self.run_callbacks("on_predict_batch_start")
            preprocessed = self.preprocess(image, path=path, **kwargs)
            raw_results = self.process(preprocessed, path=path, **kwargs)
            self.raw_results = self._as_result_list(raw_results)
            processed = self.postprocess(raw_results, image=image, path=path, **kwargs)
            self.results = self._as_result_list(processed)
            self.seen = len(self.results)
            self.run_callbacks("on_predict_postprocess_end")
            self.run_callbacks("on_predict_end")
            return processed

    def stream_inference(self, source, **kwargs):
        """Stream detector outputs over any supported BoxMOT source."""
        batch_size = max(int(kwargs.pop("batch", self.batch_size)), 1)
        vid_stride = max(int(kwargs.pop("vid_stride", self.vid_stride)), 1)

        with self._lock:
            self.stream = True
            self.seen = 0
            self.setup_source(source, batch=batch_size, vid_stride=vid_stride)
            self.run_callbacks("on_predict_start")
            try:
                for paths, frames in self.dataset:
                    self.batch = (paths, frames)
                    self.run_callbacks("on_predict_batch_start")
                    preprocessed = self.preprocess(self._batch_input(frames), paths=paths, **kwargs)
                    raw_results = self.process(preprocessed, paths=paths, **kwargs)
                    self.raw_results = self._as_result_list(raw_results)
                    processed = self.postprocess(raw_results, frames=frames, paths=paths, **kwargs)
                    self.results = self._as_result_list(processed)
                    self.run_callbacks("on_predict_postprocess_end")
                    for result in self.results:
                        self.seen += 1
                        yield result
            finally:
                self.run_callbacks("on_predict_end")

    def predict_cli(self, source, **kwargs) -> None:
        """Consume streaming inference without accumulating outputs in memory."""
        for _ in self.stream_inference(source, **kwargs):
            pass

    def predict(self, source, **kwargs):
        """Run detector inference on a source and return detections."""
        return self(source, **kwargs)

    def __call__(self, source, stream: bool = False, **kwargs):
        if stream:
            return self.stream_inference(source, **kwargs)
        if _is_single_inference_source(source):
            return self._predict_single(source, **kwargs)
        return list(self.stream_inference(source, **kwargs))
__all__ = (
    "BaseDetectorBackend",
    "Detector",
    "Detections",
    "load_weights",
    "resolve_image",
)
