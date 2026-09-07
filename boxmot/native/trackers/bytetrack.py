"""Low-level typed-v2 binding for the native ByteTrack library."""

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
_TRACKER_NAME = "bytetrack"
_DISPLAY_NAME = "ByteTrack"


def ensure_bytetrack_cpp_library(force_rebuild: bool = False) -> Path:
    return _common.ensure_tracker_library(
        tracker_name=_TRACKER_NAME,
        display_name=_DISPLAY_NAME,
        build_lock=_BUILD_LOCK,
        force_rebuild=force_rebuild,
    )


class _ByteTrackCConfig(ctypes.Structure):
    _fields_ = [
        ("min_conf", ctypes.c_float),
        ("track_thresh", ctypes.c_float),
        ("match_thresh", ctypes.c_float),
        ("track_buffer", ctypes.c_int),
        ("frame_rate", ctypes.c_int),
        ("max_obs", ctypes.c_int),
        ("asso_func", ctypes.c_char_p),
    ]


class ByteTrackLibrary:
    def __init__(self, library_path: Path) -> None:
        if sys.platform == "darwin":
            os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        self.library_path = Path(library_path)
        self._library = ctypes.CDLL(str(self.library_path))
        self._library.boxmot_bytetrack_create.argtypes = [ctypes.POINTER(_ByteTrackCConfig)]
        self._library.boxmot_bytetrack_create.restype = ctypes.c_void_p
        self._library.boxmot_bytetrack_destroy.argtypes = [ctypes.c_void_p]
        self._library.boxmot_bytetrack_destroy.restype = None
        self._library.boxmot_bytetrack_reset.argtypes = [ctypes.c_void_p]
        self._library.boxmot_bytetrack_reset.restype = ctypes.c_int
        self._update, self._result_free = _common.configure_update_v2(self._library, prefix=_TRACKER_NAME)
        self._library.boxmot_bytetrack_last_error.argtypes = []
        self._library.boxmot_bytetrack_last_error.restype = ctypes.c_char_p

    def _last_error(self) -> str:
        raw = self._library.boxmot_bytetrack_last_error()
        return "Unknown native ByteTrack error." if raw is None else raw.decode("utf-8", errors="replace")

    def create(self, cfg: Mapping[str, Any]):
        c_cfg = _ByteTrackCConfig(
            min_conf=float(cfg["min_conf"]),
            track_thresh=float(cfg["track_thresh"]),
            match_thresh=float(cfg["match_thresh"]),
            track_buffer=int(cfg["track_buffer"]),
            frame_rate=int(cfg["frame_rate"]),
            max_obs=int(cfg.get("max_obs", 50)),
            asso_func=str(cfg["asso_func"]).encode(),
        )
        handle = self._library.boxmot_bytetrack_create(ctypes.byref(c_cfg))
        if not handle:
            raise RuntimeError(self._last_error())
        return handle

    def destroy(self, handle) -> None:
        self._library.boxmot_bytetrack_destroy(handle)

    def reset(self, handle) -> None:
        if self._library.boxmot_bytetrack_reset(handle) == 0:
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


def get_bytetrack_library() -> ByteTrackLibrary:
    global _LIBRARY
    with _LIBRARY_LOCK:
        if _LIBRARY is None:
            _LIBRARY = ByteTrackLibrary(ensure_bytetrack_cpp_library())
        return _LIBRARY


__all__ = ("ByteTrackLibrary", "ensure_bytetrack_cpp_library", "get_bytetrack_library")
