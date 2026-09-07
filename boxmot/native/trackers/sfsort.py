"""Low-level typed-v2 binding for the native SFSORT library."""

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
_TRACKER_NAME = "sfsort"
_DISPLAY_NAME = "SFSORT"


def ensure_sfsort_cpp_library(force_rebuild: bool = False) -> Path:
    return _common.ensure_tracker_library(
        tracker_name=_TRACKER_NAME,
        display_name=_DISPLAY_NAME,
        build_lock=_BUILD_LOCK,
        force_rebuild=force_rebuild,
    )


class _SFSORTCConfig(ctypes.Structure):
    _fields_ = [
        ("high_th", ctypes.c_float),
        ("match_th_first", ctypes.c_float),
        ("new_track_th", ctypes.c_float),
        ("low_th", ctypes.c_float),
        ("match_th_second", ctypes.c_float),
        ("dynamic_tuning", ctypes.c_int),
        ("cth", ctypes.c_float),
        ("high_th_m", ctypes.c_float),
        ("new_track_th_m", ctypes.c_float),
        ("match_th_first_m", ctypes.c_float),
        ("obb_theta_damping", ctypes.c_float),
        ("marginal_timeout", ctypes.c_int),
        ("central_timeout", ctypes.c_int),
        ("frame_width", ctypes.c_int),
        ("frame_height", ctypes.c_int),
        ("horizontal_margin", ctypes.c_int),
        ("vertical_margin", ctypes.c_int),
        ("frame_rate", ctypes.c_int),
        ("max_obs", ctypes.c_int),
        ("asso_func", ctypes.c_char_p),
    ]


class SFSORTLibrary:
    def __init__(self, library_path: Path) -> None:
        if sys.platform == "darwin":
            os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        self.library_path = Path(library_path)
        self._library = ctypes.CDLL(str(self.library_path))
        self._library.boxmot_sfsort_create.argtypes = [ctypes.POINTER(_SFSORTCConfig)]
        self._library.boxmot_sfsort_create.restype = ctypes.c_void_p
        self._library.boxmot_sfsort_destroy.argtypes = [ctypes.c_void_p]
        self._library.boxmot_sfsort_destroy.restype = None
        self._library.boxmot_sfsort_reset.argtypes = [ctypes.c_void_p]
        self._library.boxmot_sfsort_reset.restype = ctypes.c_int
        self._update, self._result_free = _common.configure_update_v2(self._library, prefix=_TRACKER_NAME)
        self._library.boxmot_sfsort_last_error.argtypes = []
        self._library.boxmot_sfsort_last_error.restype = ctypes.c_char_p

    def _last_error(self) -> str:
        raw = self._library.boxmot_sfsort_last_error()
        return "Unknown native SFSORT error." if raw is None else raw.decode("utf-8", errors="replace")

    def create(self, cfg: Mapping[str, Any]):
        c_cfg = _SFSORTCConfig(
            high_th=float(cfg["high_th"]),
            match_th_first=float(cfg["match_th_first"]),
            new_track_th=float(cfg["new_track_th"]),
            low_th=float(cfg["low_th"]),
            match_th_second=float(cfg["match_th_second"]),
            dynamic_tuning=int(bool(cfg["dynamic_tuning"])),
            cth=float(cfg["cth"]),
            high_th_m=float(cfg["high_th_m"]),
            new_track_th_m=float(cfg["new_track_th_m"]),
            match_th_first_m=float(cfg["match_th_first_m"]),
            obb_theta_damping=float(cfg["obb_theta_damping"]),
            marginal_timeout=int(cfg["marginal_timeout"]),
            central_timeout=int(cfg["central_timeout"]),
            frame_width=int(cfg["frame_width"]),
            frame_height=int(cfg["frame_height"]),
            horizontal_margin=int(cfg["horizontal_margin"]),
            vertical_margin=int(cfg["vertical_margin"]),
            frame_rate=int(cfg["frame_rate"]),
            max_obs=int(cfg["max_obs"]),
            asso_func=str(cfg["asso_func"]).encode(),
        )
        handle = self._library.boxmot_sfsort_create(ctypes.byref(c_cfg))
        if not handle:
            raise RuntimeError(self._last_error())
        return handle

    def destroy(self, handle) -> None:
        self._library.boxmot_sfsort_destroy(handle)

    def reset(self, handle) -> None:
        if self._library.boxmot_sfsort_reset(handle) == 0:
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


def get_sfsort_library() -> SFSORTLibrary:
    global _LIBRARY
    with _LIBRARY_LOCK:
        if _LIBRARY is None:
            _LIBRARY = SFSORTLibrary(ensure_sfsort_cpp_library())
        return _LIBRARY


__all__ = ("SFSORTLibrary", "ensure_sfsort_cpp_library", "get_sfsort_library")
