"""Low-level typed-v2 binding for the native OcSort library."""

from __future__ import annotations

import ctypes
import os
import sys
import threading
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from boxmot.native.trackers import _common

_BUILD_LOCK = threading.Lock()
_LIBRARY_LOCK = threading.Lock()
_LIBRARY = None
_TRACKER_NAME = "ocsort"
_DISPLAY_NAME = "OcSort"


def ensure_ocsort_cpp_library(force_rebuild: bool = False) -> Path:
    return _common.ensure_tracker_library(
        tracker_name=_TRACKER_NAME,
        display_name=_DISPLAY_NAME,
        build_lock=_BUILD_LOCK,
        force_rebuild=force_rebuild,
    )


class _OCSortCConfig(ctypes.Structure):
    _fields_ = [
        ("min_conf", ctypes.c_float),
        ("det_thresh", ctypes.c_float),
        ("iou_threshold", ctypes.c_float),
        ("max_age", ctypes.c_int),
        ("min_hits", ctypes.c_int),
        ("delta_t", ctypes.c_int),
        ("use_byte", ctypes.c_int),
        ("inertia", ctypes.c_float),
        ("q_xy_scaling", ctypes.c_float),
        ("q_s_scaling", ctypes.c_float),
        ("max_obs", ctypes.c_int),
        ("asso_func", ctypes.c_char_p),
    ]


class OcSortLibrary:
    def __init__(self, library_path: Path) -> None:
        if sys.platform == "darwin":
            os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        self.library_path = Path(library_path)
        self._library = ctypes.CDLL(str(self.library_path))
        self._library.boxmot_ocsort_create.argtypes = [ctypes.POINTER(_OCSortCConfig)]
        self._library.boxmot_ocsort_create.restype = ctypes.c_void_p
        self._library.boxmot_ocsort_destroy.argtypes = [ctypes.c_void_p]
        self._library.boxmot_ocsort_destroy.restype = None
        self._library.boxmot_ocsort_reset.argtypes = [ctypes.c_void_p]
        self._library.boxmot_ocsort_reset.restype = ctypes.c_int
        self._update, self._result_free = _common.configure_update_v2(self._library, prefix=_TRACKER_NAME)
        self._library.boxmot_ocsort_last_error.argtypes = []
        self._library.boxmot_ocsort_last_error.restype = ctypes.c_char_p

    def _last_error(self) -> str:
        raw = self._library.boxmot_ocsort_last_error()
        return "Unknown native OcSort error." if raw is None else raw.decode("utf-8", errors="replace")

    def create(self, cfg: Mapping[str, Any]):
        c_cfg = _OCSortCConfig(
            min_conf=float(cfg["min_conf"]),
            det_thresh=float(cfg["det_thresh"]),
            iou_threshold=float(cfg["iou_threshold"]),
            max_age=int(cfg["max_age"]),
            min_hits=int(cfg["min_hits"]),
            delta_t=int(cfg["delta_t"]),
            use_byte=int(bool(cfg["use_byte"])),
            inertia=float(cfg["inertia"]),
            q_xy_scaling=float(cfg["q_xy_scaling"]),
            q_s_scaling=float(cfg["q_s_scaling"]),
            max_obs=int(cfg["max_obs"]),
            asso_func=str(cfg["asso_func"]).encode(),
        )
        handle = self._library.boxmot_ocsort_create(ctypes.byref(c_cfg))
        if not handle:
            raise RuntimeError(self._last_error())
        return handle

    def destroy(self, handle) -> None:
        self._library.boxmot_ocsort_destroy(handle)

    def reset(self, handle) -> None:
        if self._library.boxmot_ocsort_reset(handle) == 0:
            raise RuntimeError(self._last_error())

    def update(
        self,
        handle,
        *,
        geometry: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        detection_indices: np.ndarray,
        embeddings: np.ndarray | None,
        image: np.ndarray | None,
    ) -> _common.NativeTrackBatch:
        return _common.call_update_v2(
            self._update,
            self._result_free,
            handle=handle,
            geometry=geometry,
            scores=scores,
            class_ids=class_ids,
            detection_indices=detection_indices,
            embeddings=embeddings,
            image=image,
            display_name=_DISPLAY_NAME,
            last_error=self._last_error,
        )


def get_ocsort_library() -> OcSortLibrary:
    global _LIBRARY
    with _LIBRARY_LOCK:
        if _LIBRARY is None:
            _LIBRARY = OcSortLibrary(ensure_ocsort_cpp_library())
        return _LIBRARY


__all__ = ("OcSortLibrary", "ensure_ocsort_cpp_library", "get_ocsort_library")
