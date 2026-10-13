"""Low-level typed-v2 binding for the native BotSort library."""

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
_TRACKER_NAME = "botsort"
_DISPLAY_NAME = "BotSort"


def ensure_botsort_cpp_library(force_rebuild: bool = False) -> Path:
    return _common.ensure_tracker_library(
        tracker_name=_TRACKER_NAME,
        display_name=_DISPLAY_NAME,
        build_lock=_BUILD_LOCK,
        force_rebuild=force_rebuild,
    )


class _BotSortCConfigV2(ctypes.Structure):
    _fields_ = [
        ("track_high_thresh", ctypes.c_float),
        ("track_low_thresh", ctypes.c_float),
        ("new_track_thresh", ctypes.c_float),
        ("max_lost_frames", ctypes.c_int),
        ("match_thresh", ctypes.c_float),
        ("proximity_thresh", ctypes.c_float),
        ("appearance_thresh", ctypes.c_float),
        ("second_match_thresh", ctypes.c_float),
        ("unconfirmed_match_thresh", ctypes.c_float),
        ("unconfirmed_emb_scale", ctypes.c_float),
        ("cmc_method", ctypes.c_char_p),
        ("fuse_first_associate", ctypes.c_int),
        ("use_embeddings", ctypes.c_int),
        ("max_obs", ctypes.c_int),
        ("asso_func", ctypes.c_char_p),
        ("centroid_max_distance", ctypes.c_double),
    ]


class BotSortLibrary:
    def __init__(self, library_path: Path) -> None:
        if sys.platform == "darwin":
            os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        self.library_path = Path(library_path)
        self._library = ctypes.CDLL(str(self.library_path))
        self._create = getattr(self._library, "boxmot_botsort_create_v2", None)
        if self._create is None:
            raise RuntimeError(
                "Native BotSort library has an incompatible configuration ABI; rebuild the native library."
            )
        self._create.argtypes = [ctypes.POINTER(_BotSortCConfigV2)]
        self._create.restype = ctypes.c_void_p
        self._library.boxmot_botsort_destroy.argtypes = [ctypes.c_void_p]
        self._library.boxmot_botsort_destroy.restype = None
        self._library.boxmot_botsort_reset.argtypes = [ctypes.c_void_p]
        self._library.boxmot_botsort_reset.restype = ctypes.c_int
        self._update, self._result_free = _common.configure_update_v2(self._library, prefix=_TRACKER_NAME)
        self._library.boxmot_botsort_last_error.argtypes = []
        self._library.boxmot_botsort_last_error.restype = ctypes.c_char_p

    def _last_error(self) -> str:
        raw = self._library.boxmot_botsort_last_error()
        return "Unknown native BotSort error." if raw is None else raw.decode("utf-8", errors="replace")

    def create(self, cfg: Mapping[str, Any]):
        cmc_method = str(cfg.get("cmc_method", "ecc")) if bool(cfg.get("use_cmc", True)) else "none"
        c_cfg = _BotSortCConfigV2(
            track_high_thresh=float(cfg["track_high_thresh"]),
            track_low_thresh=float(cfg["track_low_thresh"]),
            new_track_thresh=float(cfg["new_track_thresh"]),
            max_lost_frames=int(cfg["max_lost_frames"]),
            match_thresh=float(cfg["match_thresh"]),
            proximity_thresh=float(cfg["proximity_thresh"]),
            appearance_thresh=float(cfg["appearance_thresh"]),
            second_match_thresh=float(cfg["second_match_thresh"]),
            unconfirmed_match_thresh=float(cfg["unconfirmed_match_thresh"]),
            unconfirmed_emb_scale=float(cfg["unconfirmed_emb_scale"]),
            cmc_method=cmc_method.encode(),
            fuse_first_associate=int(bool(cfg["fuse_first_associate"])),
            use_embeddings=int(bool(cfg["use_embeddings"])),
            max_obs=int(cfg.get("max_obs", 50)),
            asso_func=str(cfg["asso_func"]).encode(),
            centroid_max_distance=float(cfg["centroid_max_distance"]),
        )
        handle = self._create(ctypes.byref(c_cfg))
        if not handle:
            raise RuntimeError(self._last_error())
        return handle

    def destroy(self, handle) -> None:
        self._library.boxmot_botsort_destroy(handle)

    def reset(self, handle) -> None:
        if self._library.boxmot_botsort_reset(handle) == 0:
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


def get_botsort_library() -> BotSortLibrary:
    global _LIBRARY
    with _LIBRARY_LOCK:
        if _LIBRARY is None:
            _LIBRARY = BotSortLibrary(ensure_botsort_cpp_library())
        return _LIBRARY


__all__ = ("BotSortLibrary", "ensure_botsort_cpp_library", "get_botsort_library")
