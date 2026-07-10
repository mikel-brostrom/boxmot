# BoxMOT AGPL-3.0 license

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from boxmot.configs import BOXMOT_DEFAULTS
from boxmot.detectors.detector import Detector as _Detector
from boxmot.reid.core import ReID
from boxmot.utils.misc import resolve_model_path


def _ensure_model_path(model_ref: str | Path) -> Path:
    path = Path(model_ref)
    if not path.suffix:
        path = path.with_suffix(".pt")
    return resolve_model_path(path)


class Detector(_Detector):
    """High-level detector runtime with user-facing constructor names.

    This wrapper keeps ``boxmot.detectors.Detector`` as the lower-level runtime
    while making ``from boxmot import Detector`` fit the public API.
    """

    def __init__(
        self,
        model: str | Path,
        *,
        device: str = BOXMOT_DEFAULTS.track.device,
        image_size=None,
        confidence: float | None = None,
        iou: float = BOXMOT_DEFAULTS.track.iou,
        classes: Any = None,
        agnostic_nms: bool = False,
        half: bool = BOXMOT_DEFAULTS.track.half,
        batch: int = 1,
        vid_stride: int = 1,
    ) -> None:
        self.half = bool(half)
        super().__init__(
            path=_ensure_model_path(model),
            device=device,
            imgsz=image_size,
            conf=confidence,
            iou=iou,
            classes=classes,
            agnostic_nms=agnostic_nms,
            batch=batch,
            vid_stride=vid_stride,
        )

    @property
    def image_size(self):
        return self.imgsz

    @property
    def confidence(self) -> float:
        return self.conf

    @staticmethod
    def _unwrap_single(result):
        if isinstance(result, list) and len(result) == 1:
            return result[0]
        return result

    def predict(self, source, **kwargs):
        """Run inference and return ``Detections`` for a single image."""
        kwargs.setdefault("as_detections", True)
        return self._unwrap_single(super().predict(source, **kwargs))

    def __call__(self, source, stream: bool = False, **kwargs):
        kwargs.setdefault("as_detections", True)
        result = super().__call__(source, stream=stream, **kwargs)
        if stream:
            return result
        return self._unwrap_single(result)


class ReIDModel:
    """High-level ReID runtime with embed/export helpers."""

    def __init__(
        self,
        weights: str | Path,
        *,
        device: str = BOXMOT_DEFAULTS.track.device,
        half: bool = BOXMOT_DEFAULTS.track.half,
        preprocess: str | None = None,
    ) -> None:
        self.runtime = ReID(
            _ensure_model_path(weights),
            device=device,
            half=half,
            preprocess_name=preprocess,
        )
        self.path = self.runtime.path
        self.weights = self.runtime.weights
        self.device = self.runtime.device
        self.half = self.runtime.half
        self.preprocess = self.runtime.preprocess_name
        self.model = self.runtime.model
        self.export_result = None

    def embed(self, source: str | Path | Any, *, boxes: Any = None) -> np.ndarray:
        """Generate embeddings for image crops or image+box detections."""
        return self.runtime(source, boxes=boxes)

    def __call__(self, source: str | Path | Any, *, boxes: Any = None) -> np.ndarray:
        return self.embed(source, boxes=boxes)

    def get_features(self, boxes: Any, image: Any) -> np.ndarray:
        """Tracker-facing feature extraction contract."""
        return self.embed(image, boxes=boxes)

    def export(
        self,
        *,
        format: str | Sequence[str] | None = None,
        include: Sequence[str] = BOXMOT_DEFAULTS.export.include,
        device: str = BOXMOT_DEFAULTS.export.device,
        half: bool = BOXMOT_DEFAULTS.export.half,
        optimize: bool = BOXMOT_DEFAULTS.export.optimize,
        dynamic: bool = True,
        simplify: bool = BOXMOT_DEFAULTS.export.simplify,
        opset: int = BOXMOT_DEFAULTS.export.opset,
        workspace: int = BOXMOT_DEFAULTS.export.workspace,
        verbose: bool = False,
        batch_size: int = BOXMOT_DEFAULTS.export.batch_size,
        imgsz=None,
        tflite_quantize: str = BOXMOT_DEFAULTS.export.tflite_quantize,
        tflite_calibration_data=None,
        tflite_calibration_samples: int = BOXMOT_DEFAULTS.export.tflite_calibration_samples,
        tflite_calibration_preprocess: str = BOXMOT_DEFAULTS.export.tflite_calibration_preprocess,
        tflite_calibration_seed: int = BOXMOT_DEFAULTS.export.tflite_calibration_seed,
        tflite_calibration_update: str = BOXMOT_DEFAULTS.export.tflite_calibration_update,
        tflite_static_activation_bits: int = BOXMOT_DEFAULTS.export.tflite_static_activation_bits,
    ) -> "ReIDModel":
        from boxmot.pipeline import BoxMOT

        result = BoxMOT(reid=self.path).export(
            format=format,
            include=include,
            device=device,
            half=half,
            optimize=optimize,
            dynamic=dynamic,
            simplify=simplify,
            opset=opset,
            workspace=workspace,
            verbose=verbose,
            batch_size=batch_size,
            imgsz=imgsz,
            tflite_quantize=tflite_quantize,
            tflite_calibration_data=tflite_calibration_data,
            tflite_calibration_samples=tflite_calibration_samples,
            tflite_calibration_preprocess=tflite_calibration_preprocess,
            tflite_calibration_seed=tflite_calibration_seed,
            tflite_calibration_update=tflite_calibration_update,
            tflite_static_activation_bits=tflite_static_activation_bits,
        )
        exported = type(self)(
            result.embedding_weights,
            device=device,
            half=result.half,
            preprocess=self.preprocess,
        )
        exported.export_result = result
        return exported


__all__ = ("Detector", "ReIDModel")
