from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np

from boxmot.native import _common
from boxmot.native._common import (  # noqa: F401
    dets_n_embs_root,
)
from boxmot.native.trackers import _common as _native_trackers

_BUILD_LOCK = threading.Lock()
_LIVE_LIBRARY_LOCK = threading.Lock()
_LIVE_LIBRARY = None
_PROGRESS_PREFIX = _common.PROGRESS_PREFIX
_NATIVE_DISPLAY_NAME = "OCSORT"


_TRACKER_NAME = "ocsort"


def _normalize_cfg_keys(cfg: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(cfg)
    if "Q_xy_scaling" in normalized and "q_xy_scaling" not in normalized:
        normalized["q_xy_scaling"] = normalized["Q_xy_scaling"]
    if "Q_s_scaling" in normalized and "q_s_scaling" not in normalized:
        normalized["q_s_scaling"] = normalized["Q_s_scaling"]
    return normalized


def _resolve_tracker_cfg(cfg_dict: dict[str, Any] | None) -> dict[str, Any]:
    resolved = _native_trackers.load_tracker_cfg("ocsort", cfg_dict)
    resolved = _normalize_cfg_keys(resolved)

    asso_func = str(resolved.get("asso_func", "iou") or "iou").lower()
    if asso_func != "iou":
        raise NotImplementedError(
            f"Native OCSORT currently supports asso_func='iou' only, got {asso_func!r}."
        )

    resolved.setdefault("min_conf", 0.1)
    resolved.setdefault("det_thresh", 0.6)
    resolved.setdefault("iou_threshold", 0.3)
    resolved.setdefault("max_age", 30)
    resolved.setdefault("min_hits", 3)
    resolved.setdefault("delta_t", 3)
    resolved.setdefault("use_byte", False)
    resolved.setdefault("inertia", 0.1)
    resolved.setdefault("q_xy_scaling", 0.01)
    resolved.setdefault("q_s_scaling", 0.0001)
    resolved.setdefault("max_obs", int(resolved["max_age"]) + 5)
    return resolved


def ensure_ocsort_cpp_executable(force_rebuild: bool = False) -> Path:
    return _native_trackers.ensure_tracker_executable(
        tracker_name=_TRACKER_NAME,
        display_name=_NATIVE_DISPLAY_NAME,
        build_lock=_BUILD_LOCK,
        force_rebuild=force_rebuild,
    )


def ensure_ocsort_cpp_library(force_rebuild: bool = False) -> Path:
    return _native_trackers.ensure_tracker_library(
        tracker_name=_TRACKER_NAME,
        display_name=_NATIVE_DISPLAY_NAME,
        build_lock=_BUILD_LOCK,
        force_rebuild=force_rebuild,
    )


class _OCSORTCConfig(ctypes.Structure):
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
    ]


class _OCSORTLiveLibrary:
    def __init__(self, library_path: Path) -> None:
        self.library_path = Path(library_path)
        if sys.platform == "darwin":
            os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        self._library = ctypes.CDLL(str(self.library_path))
        self._configure_functions()

    def _configure_functions(self) -> None:
        self._library.boxmot_ocsort_create.argtypes = [ctypes.POINTER(_OCSORTCConfig)]
        self._library.boxmot_ocsort_create.restype = ctypes.c_void_p
        self._library.boxmot_ocsort_destroy.argtypes = [ctypes.c_void_p]
        self._library.boxmot_ocsort_destroy.restype = None
        self._library.boxmot_ocsort_reset.argtypes = [ctypes.c_void_p]
        self._library.boxmot_ocsort_reset.restype = ctypes.c_int
        self._library.boxmot_ocsort_update.argtypes = _native_trackers.LIVE_UPDATE_ARGTYPES
        self._library.boxmot_ocsort_update.restype = ctypes.c_int
        self._library.boxmot_ocsort_last_error.argtypes = []
        self._library.boxmot_ocsort_last_error.restype = ctypes.c_char_p

    def _last_error(self) -> str:
        raw = self._library.boxmot_ocsort_last_error()
        if raw is None:
            return "Unknown native OCSORT error."
        return raw.decode("utf-8", errors="replace") or "Unknown native OCSORT error."

    def create(self, cfg: dict[str, Any]):
        c_cfg = _OCSORTCConfig(
            min_conf=float(cfg["min_conf"]),
            det_thresh=float(cfg["det_thresh"]),
            iou_threshold=float(cfg.get("iou_threshold", 0.3)),
            max_age=int(cfg["max_age"]),
            min_hits=int(cfg["min_hits"]),
            delta_t=int(cfg["delta_t"]),
            use_byte=int(bool(cfg.get("use_byte", False))),
            inertia=float(cfg.get("inertia", 0.1)),
            q_xy_scaling=float(cfg.get("q_xy_scaling", 0.01)),
            q_s_scaling=float(cfg.get("q_s_scaling", 0.0001)),
            max_obs=int(cfg.get("max_obs", int(cfg["max_age"]) + 5)),
        )
        handle = self._library.boxmot_ocsort_create(ctypes.byref(c_cfg))
        if not handle:
            raise RuntimeError(self._last_error())
        return handle

    def destroy(self, handle) -> None:
        if handle:
            self._library.boxmot_ocsort_destroy(handle)

    def reset(self, handle) -> None:
        if self._library.boxmot_ocsort_reset(handle) == 0:
            raise RuntimeError(self._last_error())

    def update(self, handle, dets: np.ndarray, img: np.ndarray) -> np.ndarray:
        return _native_trackers.call_update(
            self._library.boxmot_ocsort_update,
            handle=handle,
            dets=dets,
            img=img,
            display_name=_NATIVE_DISPLAY_NAME,
            last_error=self._last_error,
        )


def _get_live_ocsort_library() -> _OCSORTLiveLibrary:
    global _LIVE_LIBRARY
    with _LIVE_LIBRARY_LOCK:
        if _LIVE_LIBRARY is None:
            _LIVE_LIBRARY = _OCSORTLiveLibrary(ensure_ocsort_cpp_library())
        return _LIVE_LIBRARY


class NativeOCSORTTracker(_native_trackers.NativeTrackerMixin):
    supports_obb = True
    tracker_name = "ocsort"
    tracker_backend = "cpp"
    provides_reid = False
    with_reid = False
    _native_display_name = _NATIVE_DISPLAY_NAME

    def __init__(
        self,
        cfg_dict: dict[str, Any] | None = None,
        *,
        reid_weights: str | Path | None = None,
        reid_preprocess: str | None = None,
        library: _OCSORTLiveLibrary | None = None,
    ) -> None:
        del reid_weights, reid_preprocess
        native_library = library if library is not None else _get_live_ocsort_library()
        self._init_native_handle(library=native_library, cfg=_resolve_tracker_cfg(cfg_dict))

    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray | None = None) -> np.ndarray:
        del embs
        det_arr = self._coerce_detections_for_mode(dets)
        return self._library.update(self._handle, det_arr, img)


def create_ocsort_live_tracker(
    cfg_dict: dict[str, Any] | None = None,
    *,
    reid_weights: str | Path | None = None,
    reid_preprocess: str | None = None,
) -> NativeOCSORTTracker:
    return NativeOCSORTTracker(cfg_dict, reid_weights=reid_weights, reid_preprocess=reid_preprocess)


def _parse_summary(stdout: str) -> dict[str, Any]:
    return _common.parse_summary(stdout, display_name=_NATIVE_DISPLAY_NAME)


def process_sequence_cpp(
    seq_name: str,
    mot_root: str,
    project_root: str,
    detector_name: str,
    reid_name: str,
    tracker_name: str,
    exp_folder: str,
    target_fps: int | None,
    cfg_dict: dict | None = None,
    dataset_name: str | None = None,
    conf_threshold: float = 0.0,
    preprocess_name: str | None = None,
    split: str | None = None,
    masks_dir: str | None = None,
    kf_tuning: dict | None = None,
    progress_queue=None,
    adaptive_kf: bool = False,
):
    del reid_name, preprocess_name
    if str(tracker_name).lower() != "ocsort":
        raise ValueError("The native cpp replay backend currently supports tracker='ocsort' only.")

    executable = ensure_ocsort_cpp_executable()
    cfg = _resolve_tracker_cfg(cfg_dict)
    det_emb_root = dets_n_embs_root(project_root, dataset_name, split=split)
    cmd = _native_trackers.build_replay_command(
        executable=executable,
        mot_root=mot_root,
        det_emb_root=det_emb_root,
        detector_name=detector_name,
        seq_name=seq_name,
        exp_folder=exp_folder,
        conf_threshold=conf_threshold,
        target_fps=target_fps,
        extra_args=[
            "--min-conf",
            str(float(cfg["min_conf"])),
            "--det-thresh",
            str(float(cfg["det_thresh"])),
            "--iou-threshold",
            str(float(cfg.get("iou_threshold", 0.3))),
            "--max-age",
            str(int(cfg["max_age"])),
            "--min-hits",
            str(int(cfg["min_hits"])),
            "--delta-t",
            str(int(cfg["delta_t"])),
            "--use-byte",
            _native_trackers.bool_arg(cfg.get("use_byte", False)),
            "--inertia",
            str(float(cfg.get("inertia", 0.1))),
            "--q-xy-scaling",
            str(float(cfg.get("q_xy_scaling", 0.01))),
            "--q-s-scaling",
            str(float(cfg.get("q_s_scaling", 0.0001))),
            "--max-obs",
            str(int(cfg.get("max_obs", int(cfg["max_age"]) + 5))),
        ],
    )

    return _native_trackers.run_replay_process(
        cmd=cmd,
        seq_name=seq_name,
        display_name=_NATIVE_DISPLAY_NAME,
        progress_queue=progress_queue,
        subprocess_module=subprocess,
    )
