"""Domain adapter for the native C++ ONNX ReID runtime."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from boxmot.native.reid.capi import get_reid_capi_library
from boxmot.reid.core.crops import coerce_boxes
from boxmot.reid.core.preprocessing import DEFAULT_PREPROCESS
from boxmot.utils import logger as LOGGER


class CppOnnxReID:
    """ReID backend that delegates feature extraction to the native C++ ABI."""

    device = "cpu"
    half = False

    def __init__(self, weights: str | Path, preprocess_name: str | None = None) -> None:
        resolved = Path(weights)
        if resolved.suffix.lower() != ".onnx":
            raise ValueError(f"CppOnnxReID requires a resolved ONNX artifact, got: {resolved}")
        if not resolved.is_file():
            raise FileNotFoundError(f"Native ReID artifact does not exist: {resolved}")

        self.weights = resolved
        self.preprocess_name = preprocess_name or DEFAULT_PREPROCESS
        self._library = get_reid_capi_library()
        self._handle = self._library.create(self.weights, self.preprocess_name)
        self._feature_dim: int | None = None
        batch, channels, height, width = self._library.input_spec(self._handle)
        if channels != 3 or height <= 0 or width <= 0:
            self.close()
            raise RuntimeError(
                "Native ReID returned an invalid ONNX input specification: "
                f"N={batch}, C={channels}, H={height}, W={width}."
            )
        self._input_shape = (height, width)
        self.input_batch_size: int | None = batch or None
        self.model = self
        LOGGER.info(
            f"CppOnnxReID using native C ABI (model={self.weights.name}, "
            f"preprocess={self.preprocess_name})"
        )

    def close(self) -> None:
        """Release the native model handle."""
        if self._handle is not None:
            self._library.destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _normalise_boxes(xyxys: np.ndarray) -> np.ndarray:
        return coerce_boxes(xyxys)

    @staticmethod
    def _normalise_image(img: np.ndarray) -> np.ndarray:
        image_arr = np.asarray(img)
        if image_arr.dtype != np.uint8:
            image_arr = image_arr.astype(np.uint8, copy=False)
        if image_arr.ndim not in {2, 3}:
            raise ValueError("Image must be a 2D or 3D uint8 array.")
        return np.ascontiguousarray(image_arr)

    @property
    def feature_dim(self) -> int:
        """Return the model output dimension reported by the native runtime."""
        if self._feature_dim is None:
            self._feature_dim = self._library.feature_dim(self._handle)
        return self._feature_dim

    @property
    def input_shape(self) -> tuple[int, int]:
        """Return the model input ``(height, width)``."""
        return self._input_shape

    def get_crops(self, xyxys: np.ndarray, img: np.ndarray) -> dict[str, int]:
        """Stage crop preparation inside the native model handle."""
        boxes = self._normalise_boxes(xyxys)
        image = self._normalise_image(img)
        self._library.preprocess(self._handle, boxes, image)
        return {"count": int(boxes.shape[0])}

    def inference_preprocess(self, payload: dict[str, int]) -> dict[str, int]:
        """Return the already-native-preprocessed call payload."""
        return payload

    def forward(self, payload: dict[str, int]) -> dict[str, int]:
        """Invoke the staged native model forward pass."""
        self._library.process(self._handle)
        return payload

    def inference_postprocess(self, payload: dict[str, int]) -> np.ndarray:
        """Copy L2-normalized features from the native output buffer."""
        count = int(payload.get("count", 0))
        if count == 0:
            return np.empty((0,), dtype=np.float32)
        out = np.empty((count, self.feature_dim), dtype=np.float32)
        self._library.postprocess(self._handle, out)
        return out

    def get_features(self, xyxys: np.ndarray, img: np.ndarray) -> np.ndarray:
        """Run native crop, inference, and postprocessing stages."""
        if xyxys is None or np.asarray(xyxys).size == 0:
            return np.array([])
        payload = self.get_crops(xyxys, img)
        payload = self.inference_preprocess(payload)
        payload = self.forward(payload)
        return self.inference_postprocess(payload)


__all__ = ("CppOnnxReID",)
