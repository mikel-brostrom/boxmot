"""Native OccluBoost configuration ABI v3 and typed update ABI v2 binding."""

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
_TRACKER_NAME = "occluboost"
_DISPLAY_NAME = "OccluBoost"


def ensure_occluboost_cpp_library(force_rebuild: bool = False) -> Path:
    return _common.ensure_tracker_library(
        tracker_name=_TRACKER_NAME,
        display_name=_DISPLAY_NAME,
        build_lock=_BUILD_LOCK,
        force_rebuild=force_rebuild,
    )


class _OccluBoostCConfigV3(ctypes.Structure):
    _fields_ = [
        ("max_age", ctypes.c_int),
        ("min_hits", ctypes.c_int),
        ("det_thresh", ctypes.c_float),
        ("iou_threshold", ctypes.c_float),
        ("min_box_area", ctypes.c_int),
        ("aspect_ratio_thresh", ctypes.c_float),
        ("lambda_iou", ctypes.c_float),
        ("lambda_mhd", ctypes.c_float),
        ("lambda_shape", ctypes.c_float),
        ("use_dlo_boost", ctypes.c_int),
        ("use_duo_boost", ctypes.c_int),
        ("dlo_boost_coef", ctypes.c_float),
        ("s_sim_corr", ctypes.c_int),
        ("use_rich_s", ctypes.c_int),
        ("use_sb", ctypes.c_int),
        ("use_vt", ctypes.c_int),
        ("use_embeddings", ctypes.c_int),
        ("cmc_method", ctypes.c_char_p),
        ("max_obs", ctypes.c_int),
        ("recovery_appearance_thresh", ctypes.c_float),
        ("recovery_iou_thresh", ctypes.c_float),
        ("recovery_max_age", ctypes.c_int),
        ("feat_alpha", ctypes.c_float),
        ("track_low_thresh", ctypes.c_float),
        ("second_iou_thresh", ctypes.c_float),
        ("second_appearance_thresh", ctypes.c_float),
        ("second_pass_max_age", ctypes.c_int),
        ("second_pass_min_hits", ctypes.c_int),
        ("use_second_pass", ctypes.c_int),
        ("new_track_thresh", ctypes.c_float),
        ("confirm_hits", ctypes.c_int),
        ("instant_confirm_thresh", ctypes.c_float),
        ("tentative_max_age", ctypes.c_int),
        ("duplicate_iou_thresh", ctypes.c_float),
        ("ams_enabled", ctypes.c_int),
        ("ams_alpha0", ctypes.c_float),
        ("ams_threshold", ctypes.c_float),
        ("ams_buffer_size", ctypes.c_int),
        ("ams_shrink_ratio", ctypes.c_float),
        ("lambda_emb_multiplier", ctypes.c_float),
        ("asso_func", ctypes.c_char_p),
        ("centroid_max_distance", ctypes.c_double),
        ("alpha_fixed_emb", ctypes.c_double),
        ("appearance_thresh", ctypes.c_double),
        ("appearance_iou_fraction", ctypes.c_double),
        ("dlo_soft_alpha", ctypes.c_double),
        ("dlo_soft_power", ctypes.c_double),
        ("dlo_threshold_start", ctypes.c_double),
        ("dlo_threshold_end", ctypes.c_double),
        ("duo_mahalanobis_thresh", ctypes.c_double),
        ("duo_iou_thresh", ctypes.c_double),
        ("track_confidence_decay", ctypes.c_double),
        ("track_confidence_warmup", ctypes.c_int),
    ]


def _build_c_config(cfg: Mapping[str, Any]) -> _OccluBoostCConfigV3:
    cmc_method = str(cfg["cmc_method"]) if bool(cfg["use_cmc"]) else "none"
    return _OccluBoostCConfigV3(
        max_age=int(cfg["max_age"]),
        min_hits=int(cfg["min_hits"]),
        det_thresh=float(cfg["det_thresh"]),
        iou_threshold=float(cfg["iou_threshold"]),
        min_box_area=int(cfg["min_box_area"]),
        aspect_ratio_thresh=float(cfg["aspect_ratio_thresh"]),
        lambda_iou=float(cfg["lambda_iou"]),
        lambda_mhd=float(cfg["lambda_mhd"]),
        lambda_shape=float(cfg["lambda_shape"]),
        use_dlo_boost=int(bool(cfg["use_dlo_boost"])),
        use_duo_boost=int(bool(cfg["use_duo_boost"])),
        dlo_boost_coef=float(cfg["dlo_boost_coef"]),
        s_sim_corr=int(bool(cfg["s_sim_corr"])),
        use_rich_s=int(bool(cfg["use_rich_s"])),
        use_sb=int(bool(cfg["use_sb"])),
        use_vt=int(bool(cfg["use_vt"])),
        use_embeddings=int(bool(cfg["use_embeddings"])),
        cmc_method=cmc_method.encode(),
        max_obs=int(cfg["max_obs"]),
        recovery_appearance_thresh=float(cfg["recovery_appearance_thresh"]),
        recovery_iou_thresh=float(cfg["recovery_iou_thresh"]),
        recovery_max_age=int(cfg["recovery_max_age"]),
        feat_alpha=float(cfg["feat_alpha"]),
        track_low_thresh=float(cfg["track_low_thresh"]),
        second_iou_thresh=float(cfg["second_iou_thresh"]),
        second_appearance_thresh=float(cfg["second_appearance_thresh"]),
        second_pass_max_age=int(cfg["second_pass_max_age"]),
        second_pass_min_hits=int(cfg["second_pass_min_hits"]),
        use_second_pass=int(bool(cfg["use_second_pass"])),
        new_track_thresh=float(cfg["new_track_thresh"]),
        confirm_hits=int(cfg["confirm_hits"]),
        instant_confirm_thresh=float(cfg["instant_confirm_thresh"]),
        tentative_max_age=int(cfg["tentative_max_age"]),
        duplicate_iou_thresh=float(cfg["duplicate_iou_thresh"]),
        ams_enabled=int(bool(cfg["ams_enabled"])),
        ams_alpha0=float(cfg["ams_alpha0"]),
        ams_threshold=float(cfg["ams_threshold"]),
        ams_buffer_size=int(cfg["ams_buffer_size"]),
        ams_shrink_ratio=float(cfg["ams_shrink_ratio"]),
        lambda_emb_multiplier=float(cfg["lambda_emb_multiplier"]),
        asso_func=str(cfg["asso_func"]).encode(),
        centroid_max_distance=float(cfg["centroid_max_distance"]),
        alpha_fixed_emb=float(cfg["alpha_fixed_emb"]),
        appearance_thresh=float(cfg["appearance_thresh"]),
        appearance_iou_fraction=float(cfg["appearance_iou_fraction"]),
        dlo_soft_alpha=float(cfg["dlo_soft_alpha"]),
        dlo_soft_power=float(cfg["dlo_soft_power"]),
        dlo_threshold_start=float(cfg["dlo_threshold_start"]),
        dlo_threshold_end=float(cfg["dlo_threshold_end"]),
        duo_mahalanobis_thresh=float(cfg["duo_mahalanobis_thresh"]),
        duo_iou_thresh=float(cfg["duo_iou_thresh"]),
        track_confidence_decay=float(cfg["track_confidence_decay"]),
        track_confidence_warmup=int(cfg["track_confidence_warmup"]),
    )


class OccluBoostLibrary:
    def __init__(self, library_path: Path) -> None:
        if sys.platform == "darwin":
            os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        self.library_path = Path(library_path)
        self._library = ctypes.CDLL(str(self.library_path))
        self._create = getattr(self._library, "boxmot_occluboost_create_v3", None)
        if self._create is None:
            raise RuntimeError(
                "Native OccluBoost library has an incompatible configuration ABI; rebuild the native library."
            )
        self._create.argtypes = [ctypes.POINTER(_OccluBoostCConfigV3)]
        self._create.restype = ctypes.c_void_p
        self._library.boxmot_occluboost_destroy.argtypes = [ctypes.c_void_p]
        self._library.boxmot_occluboost_destroy.restype = None
        self._library.boxmot_occluboost_reset.argtypes = [ctypes.c_void_p]
        self._library.boxmot_occluboost_reset.restype = ctypes.c_int
        self._update, self._result_free = _common.configure_update_v2(self._library, prefix=_TRACKER_NAME)
        self._library.boxmot_occluboost_last_error.argtypes = []
        self._library.boxmot_occluboost_last_error.restype = ctypes.c_char_p

    def _last_error(self) -> str:
        raw = self._library.boxmot_occluboost_last_error()
        return "Unknown native OccluBoost error." if raw is None else raw.decode("utf-8", errors="replace")

    def create(self, cfg: Mapping[str, Any]):
        c_cfg = _build_c_config(cfg)
        handle = self._create(ctypes.byref(c_cfg))
        if not handle:
            raise RuntimeError(self._last_error())
        return handle

    def destroy(self, handle) -> None:
        self._library.boxmot_occluboost_destroy(handle)

    def reset(self, handle) -> None:
        if self._library.boxmot_occluboost_reset(handle) == 0:
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


def get_occluboost_library() -> OccluBoostLibrary:
    global _LIBRARY
    with _LIBRARY_LOCK:
        if _LIBRARY is None:
            _LIBRARY = OccluBoostLibrary(ensure_occluboost_cpp_library())
        return _LIBRARY


__all__ = (
    "OccluBoostLibrary",
    "ensure_occluboost_cpp_library",
    "get_occluboost_library",
)
