"""Python wrapper around the native ``reid_capi`` shared library.

Exposes a ``CppOnnxReID`` class that mimics the public surface of
``boxmot.reid.backends.base_backend.BaseModelBackend`` (just ``get_features``)
so it can be plugged into :class:`boxmot.engine.tracking.inference.DetectorReIDPipeline`
in place of the Python ONNXRuntime backend. This is what makes
``--tracker-backend cpp`` produce its embedding cache via the exact same C++
ONNX inference path that the C++ trackers use at replay time.

Notes
-----
* Dynamic-batch ONNX Runtime graphs process all staged crops together. Fixed
  graphs use padded chunks, while the OpenCV DNN fallback remains crop-by-crop
  for dynamic graphs whose heads may not support ``N > 1`` reliably.
* OBB detections (5-column ``cxcywh-theta``) stay oriented across the C ABI and
  use the same rectified crop implementation as the native trackers.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
from pathlib import Path

import numpy as np

from boxmot.core.box_schema import OBB_SCHEMA
from boxmot.native import _common
from boxmot.reid.core.crops import coerce_boxes
from boxmot.utils import logger as LOGGER

_BUILD_LOCK = threading.Lock()
_LIBRARY_LOCK = threading.Lock()
_LIBRARY = None


# ---------------------------------------------------------------------------
# Build / load
# ---------------------------------------------------------------------------

_TARGET_NAME = "base"  # reid_capi is built from native/cpp/trackers/base
_CMAKE_TARGET = "reid_capi"


def _library_name() -> str:
    if os.name == "nt":
        return "reid_capi.dll"
    if sys.platform == "darwin":
        return "reid_capi.dylib"
    return "reid_capi.so"


def _candidate_libraries() -> list[Path]:
    name = _library_name()
    return _common.installed_library_candidates(_TARGET_NAME, name) + _common.build_library_candidates(
        _TARGET_NAME, name
    )


def ensure_reid_capi_library(force_rebuild: bool = False) -> Path:
    """Return a source-fresh native ReID C ABI library.

    Editable builds use the shared source fingerprint and artifact stamp;
    packaged libraries beside the native sources remain trusted in installed
    wheels, where rebuilding may be unavailable or undesirable.
    """
    return _common.build_native_target(
        tracker_name=_TARGET_NAME,
        display_name="ReID C ABI",
        target=_CMAKE_TARGET,
        candidates=_candidate_libraries(),
        force_rebuild=force_rebuild,
        not_found_message="Native ReID C ABI build succeeded but the shared library was not found.",
        build_lock=_BUILD_LOCK,
    )


class _ReidLibrary:
    """Thin ctypes binding around ``reid_capi``."""

    def __init__(self, library_path: Path) -> None:
        self.library_path = Path(library_path)
        # Homebrew OpenCV pulls in OpenBLAS / libomp which conflicts with the
        # libomp PyTorch loads first on macOS. Allow them to coexist in-process.
        if sys.platform == "darwin":
            os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        self._library = ctypes.CDLL(str(self.library_path))
        self._configure()

    def _configure(self) -> None:
        self._library.boxmot_reid_capi_create.argtypes = [
            ctypes.c_char_p,  # model_path
            ctypes.c_char_p,  # preprocess
            ctypes.POINTER(ctypes.c_void_p),  # out_handle
        ]
        self._library.boxmot_reid_capi_create.restype = ctypes.c_int

        self._library.boxmot_reid_capi_destroy.argtypes = [ctypes.c_void_p]
        self._library.boxmot_reid_capi_destroy.restype = None

        self._library.boxmot_reid_capi_feature_dim.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
        ]
        self._library.boxmot_reid_capi_feature_dim.restype = ctypes.c_int

        self._library.boxmot_reid_capi_input_spec.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),  # batch (0 == dynamic)
            ctypes.POINTER(ctypes.c_int),  # channels
            ctypes.POINTER(ctypes.c_int),  # height
            ctypes.POINTER(ctypes.c_int),  # width
        ]
        self._library.boxmot_reid_capi_input_spec.restype = ctypes.c_int

        self._library.boxmot_reid_capi_compute_features.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.c_void_p,  # boxes_xyxy
            ctypes.c_int,  # n_boxes
            ctypes.c_void_p,  # image_data
            ctypes.c_int,  # image_rows
            ctypes.c_int,  # image_cols
            ctypes.c_int,  # image_channels
            ctypes.c_void_p,  # out_features
            ctypes.c_int,  # out_capacity_floats
        ]
        self._library.boxmot_reid_capi_compute_features.restype = ctypes.c_int

        self._library.boxmot_reid_capi_preprocess.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.c_void_p,  # boxes_xyxy
            ctypes.c_int,  # n_boxes
            ctypes.c_void_p,  # image_data
            ctypes.c_int,  # image_rows
            ctypes.c_int,  # image_cols
            ctypes.c_int,  # image_channels
        ]
        self._library.boxmot_reid_capi_preprocess.restype = ctypes.c_int

        self._library.boxmot_reid_capi_preprocess_obb.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.c_void_p,  # boxes_xywha
            ctypes.c_int,  # n_boxes
            ctypes.c_void_p,  # image_data
            ctypes.c_int,  # image_rows
            ctypes.c_int,  # image_cols
            ctypes.c_int,  # image_channels
        ]
        self._library.boxmot_reid_capi_preprocess_obb.restype = ctypes.c_int

        self._library.boxmot_reid_capi_process.argtypes = [ctypes.c_void_p]
        self._library.boxmot_reid_capi_process.restype = ctypes.c_int

        self._library.boxmot_reid_capi_postprocess.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.c_void_p,  # out_features
            ctypes.c_int,  # out_capacity_floats
        ]
        self._library.boxmot_reid_capi_postprocess.restype = ctypes.c_int

        self._library.boxmot_reid_capi_last_error.argtypes = []
        self._library.boxmot_reid_capi_last_error.restype = ctypes.c_char_p

    def last_error(self) -> str:
        raw = self._library.boxmot_reid_capi_last_error()
        if raw is None:
            return "Unknown native ReID error."
        return raw.decode("utf-8", errors="replace") or "Unknown native ReID error."

    def create(self, model_path: Path, preprocess_name: str | None) -> ctypes.c_void_p:
        handle = ctypes.c_void_p(0)
        from boxmot.reid.core.preprocessing import DEFAULT_PREPROCESS

        ok = self._library.boxmot_reid_capi_create(
            str(model_path).encode("utf-8"),
            (preprocess_name or DEFAULT_PREPROCESS).encode("utf-8"),
            ctypes.byref(handle),
        )
        if ok == 0 or not handle.value:
            raise RuntimeError(self.last_error())
        return handle

    def destroy(self, handle: ctypes.c_void_p) -> None:
        if handle and handle.value:
            self._library.boxmot_reid_capi_destroy(handle)

    def feature_dim(self, handle: ctypes.c_void_p) -> int:
        out_dim = ctypes.c_int(0)
        ok = self._library.boxmot_reid_capi_feature_dim(handle, ctypes.byref(out_dim))
        if ok == 0:
            raise RuntimeError(self.last_error())
        return int(out_dim.value)

    def input_spec(self, handle: ctypes.c_void_p) -> tuple[int, int, int, int]:
        batch = ctypes.c_int(0)
        channels = ctypes.c_int(0)
        height = ctypes.c_int(0)
        width = ctypes.c_int(0)
        ok = self._library.boxmot_reid_capi_input_spec(
            handle,
            ctypes.byref(batch),
            ctypes.byref(channels),
            ctypes.byref(height),
            ctypes.byref(width),
        )
        if ok == 0:
            raise RuntimeError(self.last_error())
        return int(batch.value), int(channels.value), int(height.value), int(width.value)

    def compute_features(
        self,
        handle: ctypes.c_void_p,
        boxes_xyxy: np.ndarray,
        image: np.ndarray,
        out_features: np.ndarray,
    ) -> None:
        n = int(boxes_xyxy.shape[0])
        ok = self._library.boxmot_reid_capi_compute_features(
            handle,
            None if n == 0 else ctypes.c_void_p(boxes_xyxy.ctypes.data),
            n,
            ctypes.c_void_p(image.ctypes.data),
            int(image.shape[0]),
            int(image.shape[1]),
            1 if image.ndim == 2 else int(image.shape[2]),
            ctypes.c_void_p(out_features.ctypes.data),
            int(out_features.size),
        )
        if ok == 0:
            raise RuntimeError(self.last_error())

    def preprocess(
        self,
        handle: ctypes.c_void_p,
        boxes: np.ndarray,
        image: np.ndarray,
    ) -> None:
        n = int(boxes.shape[0])
        preprocess = (
            self._library.boxmot_reid_capi_preprocess_obb
            if boxes.shape[1] == OBB_SCHEMA.geometry_cols
            else self._library.boxmot_reid_capi_preprocess
        )
        ok = preprocess(
            handle,
            None if n == 0 else ctypes.c_void_p(boxes.ctypes.data),
            n,
            ctypes.c_void_p(image.ctypes.data),
            int(image.shape[0]),
            int(image.shape[1]),
            1 if image.ndim == 2 else int(image.shape[2]),
        )
        if ok == 0:
            raise RuntimeError(self.last_error())

    def process(self, handle: ctypes.c_void_p) -> None:
        ok = self._library.boxmot_reid_capi_process(handle)
        if ok == 0:
            raise RuntimeError(self.last_error())

    def postprocess(self, handle: ctypes.c_void_p, out_features: np.ndarray) -> None:
        ok = self._library.boxmot_reid_capi_postprocess(
            handle,
            ctypes.c_void_p(out_features.ctypes.data),
            int(out_features.size),
        )
        if ok == 0:
            raise RuntimeError(self.last_error())


def _get_library() -> _ReidLibrary:
    global _LIBRARY
    with _LIBRARY_LOCK:
        if _LIBRARY is None:
            _LIBRARY = _ReidLibrary(ensure_reid_capi_library())
        return _LIBRARY


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class CppOnnxReID:
    """ReID backend that delegates feature extraction to the native C++ path.

    The instance is its own ``model`` so it can be wrapped by ``TimedReIDModel``
    the same way a Python ``ReID().model`` is. Mirrors the
    ``BaseModelBackend`` surface (``get_crops`` / ``inference_preprocess`` /
    ``forward`` / ``inference_postprocess``) so the staged C ABI symbols are
    invoked separately and timing instrumentation can attribute work to the
    ``preprocess`` / ``process`` / ``postprocess`` buckets.
    """

    # ``BaseModelBackend``-shaped attributes (used by Python ReID code paths
    # that introspect device / dtype / input_shape on the wrapped backend).
    device = "cpu"
    half = False

    def __init__(self, weights, preprocess_name: str | None = None) -> None:
        # Auto-export ``.pt`` to ``.onnx`` if needed (mirrors live native trackers).
        resolved = _common.ensure_native_reid_model_path(
            weights,
            display_name="ReID",
        )
        if resolved is None:
            raise ValueError("CppOnnxReID requires a ReID weights path.")
        if resolved.suffix.lower() != ".onnx":
            raise RuntimeError(f"CppOnnxReID requires an ONNX model after auto-export. Got: {resolved}")

        self.weights = Path(resolved)
        from boxmot.reid.core.preprocessing import DEFAULT_PREPROCESS

        self.preprocess_name = preprocess_name or DEFAULT_PREPROCESS

        self._library = _get_library()
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

        # ``DetectorReIDPipeline`` wraps ``backend.model`` via ``TimedReIDModel``;
        # for the Python backends ``ReID().model`` is a ``BaseModelBackend``
        # exposing ``get_features``. Mirror that surface.
        self.model = self

        LOGGER.info(f"CppOnnxReID using native C ABI (model={self.weights.name}, preprocess={self.preprocess_name})")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._handle is not None:
            self._library.destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Public API (matches BaseModelBackend)
    # ------------------------------------------------------------------

    @property
    def feature_dim(self) -> int:
        if self._feature_dim is None:
            self._feature_dim = self._library.feature_dim(self._handle)
        return self._feature_dim

    @property
    def input_shape(self) -> tuple[int, int]:
        return self._input_shape

    def get_crops(self, xyxys: np.ndarray, img: np.ndarray):
        """Stage 1: stage the (N, 3, H, W) crop blob inside the native handle.

        Returns an opaque payload that ``forward`` consumes. The crop tensor
        itself never crosses the FFI boundary; the payload only carries the
        per-call shape so ``forward`` can size its output buffer.
        """
        boxes = self._normalise_boxes(xyxys)
        image = self._normalise_image(img)
        self._library.preprocess(self._handle, boxes, image)
        return {"count": int(boxes.shape[0])}

    def inference_preprocess(self, payload):
        # The native side has already produced its (NHWC/NCHW) layout in
        # ``Preprocess``; nothing left to do here.
        return payload

    def forward(self, payload):
        """Stage 2: invoke the native model forward pass on the staged blob."""
        self._library.process(self._handle)
        return payload

    def inference_postprocess(self, payload):
        """Stage 3: copy the L2-normalised features back into a numpy array."""
        count = int(payload.get("count", 0))
        if count == 0:
            return np.empty((0,), dtype=np.float32)
        feature_dim = self.feature_dim
        out = np.empty((count, feature_dim), dtype=np.float32)
        self._library.postprocess(self._handle, out)
        return out

    def get_features(self, xyxys: np.ndarray, img: np.ndarray) -> np.ndarray:
        """Single-shot composite for callers that don't need staged timing."""
        if xyxys is None or np.asarray(xyxys).size == 0:
            return np.array([])
        payload = self.get_crops(xyxys, img)
        payload = self.inference_preprocess(payload)
        payload = self.forward(payload)
        return self.inference_postprocess(payload)


__all__ = [
    "CppOnnxReID",
    "ensure_reid_capi_library",
]
