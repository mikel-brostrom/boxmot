"""Low-level ctypes binding for the native ``reid_capi`` library.

This module deliberately knows nothing about BoxMOT's appearance-encoder
contract.  The domain-facing adapter lives in :mod:`boxmot.reid.backends.native`.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
from pathlib import Path

import numpy as np

from boxmot.native import _common

_BUILD_LOCK = threading.Lock()
_LIBRARY_LOCK = threading.Lock()
_LIBRARY = None
_OBB_GEOMETRY_COLUMNS = 5


# ---------------------------------------------------------------------------
# Build / load
# ---------------------------------------------------------------------------

_TARGET_NAME = "reid"
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


class ReIDLibrary:
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

    def create(self, model_path: Path, preprocess_name: str) -> ctypes.c_void_p:
        handle = ctypes.c_void_p(0)
        ok = self._library.boxmot_reid_capi_create(
            str(model_path).encode("utf-8"),
            preprocess_name.encode("utf-8"),
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
            if boxes.shape[1] == _OBB_GEOMETRY_COLUMNS
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


def get_reid_capi_library() -> ReIDLibrary:
    """Return the process-wide loaded ReID C-ABI binding."""
    global _LIBRARY
    with _LIBRARY_LOCK:
        if _LIBRARY is None:
            _LIBRARY = ReIDLibrary(ensure_reid_capi_library())
        return _LIBRARY


__all__ = ("ReIDLibrary", "ensure_reid_capi_library", "get_reid_capi_library")
