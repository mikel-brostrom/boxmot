"""Python wrapper around the native ``reid_capi`` shared library.

Exposes a ``CppOnnxReID`` class that mimics the public surface of
``boxmot.reid.backends.base_backend.BaseModelBackend`` (just ``get_features``)
so it can be plugged into :class:`boxmot.engine.inference.DetectorReIDPipeline`
in place of the Python ONNXRuntime backend. This is what makes
``--tracker-backend cpp`` produce its embedding cache via the exact same C++
ONNX inference path that the C++ trackers use at replay time.

Notes
-----
* The native side currently runs a **single** crop per ORT call (mirroring
  the live tracker behavior). For embedding-cache generation this is slower
  than the bucketed Python path on small per-frame det counts, but the cost is
  paid once and cached to ``dets_n_embs/.../embs/<reid>/<preprocess>/<seq>.npy``.
* OBB detections (5-column ``cxcywh-theta``) are converted to enclosing AABB
  rects on the Python side before being handed to the C ABI, matching what the
  C++ trackers do internally.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

from boxmot.native import _common
from boxmot.utils import logger as LOGGER

_BUILD_LOCK = threading.Lock()
_LIBRARY_LOCK = threading.Lock()
_LIBRARY = None


# ---------------------------------------------------------------------------
# Build / load
# ---------------------------------------------------------------------------

_TARGET_NAME = "base"  # reid_capi is built from native/trackers/base


def _source_dir() -> Path:
    return _common.tracker_source_dir(_TARGET_NAME)


def _build_dir() -> Path:
    return _common.tracker_build_dir(_TARGET_NAME)


def _library_name() -> str:
    if os.name == "nt":
        return "reid_capi.dll"
    if sys.platform == "darwin":
        return "reid_capi.dylib"
    return "reid_capi.so"


def _candidate_libraries() -> list[Path]:
    name = _library_name()
    return (
        _common.installed_library_candidates(_TARGET_NAME, name)
        + _common.build_library_candidates(_TARGET_NAME, name)
    )


def _build_library() -> Path:
    """Configure + build the ``reid_capi`` shared library if it's missing."""
    with _BUILD_LOCK:
        for candidate in _candidate_libraries():
            if candidate.exists():
                return candidate

        source_dir = _source_dir()
        build_dir = _build_dir()
        build_dir.mkdir(parents=True, exist_ok=True)

        # Cross-process lock: serialize CMake invocations from concurrent
        # worker subprocesses so they don't corrupt each other's build cache.
        with _common._cross_process_build_lock(build_dir):
            for candidate in _candidate_libraries():
                if candidate.exists():
                    return candidate

            configure_cmd = [
                "cmake",
                "-S",
                str(source_dir),
                "-B",
                str(build_dir),
                "-DCMAKE_BUILD_TYPE=Release",
            ]
            # Stream output live so the user sees progress.
            print("[boxmot build] reid: configuring...", flush=True)
            configure = subprocess.run(configure_cmd, check=False)
            if configure.returncode != 0:
                raise RuntimeError(
                    "Failed to configure native ReID C ABI.\n"
                    "Requirements: CMake 3.16+, OpenCV 4.x, Eigen3 3.3+, ONNX Runtime.\n"
                    f"Command: {' '.join(configure_cmd)}"
                )

            build_cmd = [
                "cmake",
                "--build",
                str(build_dir),
                "--config",
                "Release",
                "--target",
                "reid_capi",
                "--parallel",
            ]
            print("[boxmot build] reid: compiling...", flush=True)
            build = subprocess.run(build_cmd, check=False)
            if build.returncode != 0:
                raise RuntimeError(
                    "Failed to build native ReID C ABI.\n"
                    "Requirements: C++17 compiler, OpenCV 4.x, Eigen3 3.3+, ONNX Runtime.\n"
                    f"Command: {' '.join(build_cmd)}"
                )

            for candidate in _candidate_libraries():
                if candidate.exists():
                    return candidate

            raise RuntimeError(
                "Native ReID C ABI build succeeded but the shared library was not found."
            )


def ensure_reid_capi_library(force_rebuild: bool = False) -> Path:
    if not force_rebuild:
        for candidate in _candidate_libraries():
            if candidate.exists():
                return candidate
    return _build_library()


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

        self._library.boxmot_reid_capi_compute_features.argtypes = [
            ctypes.c_void_p,            # handle
            ctypes.c_void_p,            # boxes_xyxy
            ctypes.c_int,               # n_boxes
            ctypes.c_void_p,            # image_data
            ctypes.c_int,               # image_rows
            ctypes.c_int,               # image_cols
            ctypes.c_int,               # image_channels
            ctypes.c_void_p,            # out_features
            ctypes.c_int,               # out_capacity_floats
        ]
        self._library.boxmot_reid_capi_compute_features.restype = ctypes.c_int

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


def _get_library() -> _ReidLibrary:
    global _LIBRARY
    with _LIBRARY_LOCK:
        if _LIBRARY is None:
            _LIBRARY = _ReidLibrary(ensure_reid_capi_library())
        return _LIBRARY


# ---------------------------------------------------------------------------
# Box helpers
# ---------------------------------------------------------------------------

def _obb_to_enclosing_xyxy(xywha: np.ndarray) -> np.ndarray:
    """Convert OBB rows (cx, cy, w, h, theta) to enclosing AABBs."""
    if xywha.size == 0:
        return np.empty((0, 4), dtype=np.float32)

    cx = xywha[:, 0]
    cy = xywha[:, 1]
    w = np.maximum(xywha[:, 2], 1e-4)
    h = np.maximum(xywha[:, 3], 1e-4)
    theta = xywha[:, 4]

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Half extents for each oriented rectangle
    half_w = w / 2.0
    half_h = h / 2.0

    # 4 corners offsets in the oriented frame: (+/-w/2, +/-h/2)
    dx = np.stack([half_w, half_w, -half_w, -half_w], axis=1)
    dy = np.stack([half_h, -half_h, half_h, -half_h], axis=1)
    rx = dx * cos_t[:, None] - dy * sin_t[:, None]
    ry = dx * sin_t[:, None] + dy * cos_t[:, None]

    xs = cx[:, None] + rx
    ys = cy[:, None] + ry
    x1 = xs.min(axis=1)
    y1 = ys.min(axis=1)
    x2 = xs.max(axis=1)
    y2 = ys.max(axis=1)
    return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class CppOnnxReID:
    """ReID backend that delegates feature extraction to the native C++ path.

    The instance is its own ``model`` so it can be wrapped by ``TimedReIDModel``
    the same way a Python ``ReID().model`` is.
    """

    def __init__(self, weights, preprocess_name: str | None = None) -> None:
        # Auto-export ``.pt`` to ``.onnx`` if needed (mirrors live native trackers).
        resolved = _common.ensure_native_reid_model_path(
            weights,
            display_name="ReID",
        )
        if resolved is None:
            raise ValueError("CppOnnxReID requires a ReID weights path.")
        if resolved.suffix.lower() != ".onnx":
            raise RuntimeError(
                "CppOnnxReID requires an ONNX model after auto-export. Got: "
                f"{resolved}"
            )

        self.weights = Path(resolved)
        from boxmot.reid.core.preprocessing import DEFAULT_PREPROCESS
        self.preprocess_name = preprocess_name or DEFAULT_PREPROCESS

        self._library = _get_library()
        self._handle = self._library.create(self.weights, self.preprocess_name)
        self._feature_dim: int | None = None

        # ``DetectorReIDPipeline`` wraps ``backend.model`` via ``TimedReIDModel``;
        # for the Python backends ``ReID().model`` is a ``BaseModelBackend``
        # exposing ``get_features``. Mirror that surface.
        self.model = self

        LOGGER.info(
            f"CppOnnxReID using native C ABI (model={self.weights.name}, "
            f"preprocess={self.preprocess_name})"
        )

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
    # Public API (matches BaseModelBackend.get_features)
    # ------------------------------------------------------------------

    @property
    def feature_dim(self) -> int:
        if self._feature_dim is None:
            self._feature_dim = self._library.feature_dim(self._handle)
        return self._feature_dim

    def get_features(self, xyxys: np.ndarray, img: np.ndarray) -> np.ndarray:
        if xyxys is None or xyxys.size == 0:
            return np.array([])

        boxes = np.asarray(xyxys, dtype=np.float32)
        if boxes.ndim == 1:
            boxes = boxes.reshape(1, -1)
        if boxes.ndim != 2 or boxes.shape[1] not in {4, 5}:
            raise ValueError(
                "CppOnnxReID expects detections as (N, 4) AABB or (N, 5) OBB, got "
                f"shape {boxes.shape}"
            )
        if boxes.shape[1] == 5:
            boxes = _obb_to_enclosing_xyxy(boxes)
        else:
            boxes = boxes[:, :4].astype(np.float32, copy=False)

        boxes = np.ascontiguousarray(boxes, dtype=np.float32)

        image_arr = np.asarray(img)
        if image_arr.dtype != np.uint8:
            image_arr = image_arr.astype(np.uint8, copy=False)
        if image_arr.ndim not in {2, 3}:
            raise ValueError("Image must be a 2D or 3D uint8 array.")
        image_arr = np.ascontiguousarray(image_arr)

        feature_dim = self.feature_dim
        n = int(boxes.shape[0])
        out = np.empty((n, feature_dim), dtype=np.float32)
        self._library.compute_features(self._handle, boxes, image_arr, out)
        return out


__all__ = [
    "CppOnnxReID",
    "ensure_reid_capi_library",
]
