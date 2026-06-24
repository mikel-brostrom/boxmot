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
_NATIVE_DISPLAY_NAME = "SFSORT"


_TRACKER_NAME = "sfsort"


def _resolve_tracker_cfg(cfg_dict: dict[str, Any] | None) -> dict[str, Any]:
    resolved = _native_trackers.load_tracker_cfg("sfsort", cfg_dict)
    resolved.setdefault("dynamic_tuning", False)
    resolved.setdefault("cth", 0.5)
    resolved.setdefault("high_th_m", 0.0)
    resolved.setdefault("new_track_th_m", 0.0)
    resolved.setdefault("match_th_first_m", 0.0)
    resolved.setdefault("obb_theta_damping", 0.8)
    resolved.setdefault("marginal_timeout", 0)
    resolved.setdefault("central_timeout", 0)
    resolved.setdefault("frame_width", 0)
    resolved.setdefault("frame_height", 0)
    resolved.setdefault("horizontal_margin", 0)
    resolved.setdefault("vertical_margin", 0)
    resolved.setdefault("frame_rate", 30)
    resolved.setdefault("max_obs", 50)
    return resolved


def ensure_sfsort_cpp_executable(force_rebuild: bool = False) -> Path:
    return _native_trackers.ensure_tracker_executable(
        tracker_name=_TRACKER_NAME,
        display_name=_NATIVE_DISPLAY_NAME,
        build_lock=_BUILD_LOCK,
        force_rebuild=force_rebuild,
    )


def ensure_sfsort_cpp_library(force_rebuild: bool = False) -> Path:
    return _native_trackers.ensure_tracker_library(
        tracker_name=_TRACKER_NAME,
        display_name=_NATIVE_DISPLAY_NAME,
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
    ]


class _SFSORTLiveLibrary:
    def __init__(self, library_path: Path) -> None:
        self.library_path = Path(library_path)
        if sys.platform == "darwin":
            os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        self._library = ctypes.CDLL(str(self.library_path))
        self._configure_functions()

    def _configure_functions(self) -> None:
        self._library.boxmot_sfsort_create.argtypes = [ctypes.POINTER(_SFSORTCConfig)]
        self._library.boxmot_sfsort_create.restype = ctypes.c_void_p
        self._library.boxmot_sfsort_destroy.argtypes = [ctypes.c_void_p]
        self._library.boxmot_sfsort_destroy.restype = None
        self._library.boxmot_sfsort_reset.argtypes = [ctypes.c_void_p]
        self._library.boxmot_sfsort_reset.restype = ctypes.c_int
        self._library.boxmot_sfsort_update.argtypes = _native_trackers.LIVE_UPDATE_ARGTYPES
        self._library.boxmot_sfsort_update.restype = ctypes.c_int
        self._library.boxmot_sfsort_last_error.argtypes = []
        self._library.boxmot_sfsort_last_error.restype = ctypes.c_char_p

    def _last_error(self) -> str:
        raw = self._library.boxmot_sfsort_last_error()
        if raw is None:
            return "Unknown native SFSORT error."
        return raw.decode("utf-8", errors="replace") or "Unknown native SFSORT error."

    def create(self, cfg: dict[str, Any]):
        c_cfg = _SFSORTCConfig(
            high_th=float(cfg["high_th"]),
            match_th_first=float(cfg["match_th_first"]),
            new_track_th=float(cfg["new_track_th"]),
            low_th=float(cfg["low_th"]),
            match_th_second=float(cfg["match_th_second"]),
            dynamic_tuning=int(bool(cfg.get("dynamic_tuning", False))),
            cth=float(cfg.get("cth", 0.5)),
            high_th_m=float(cfg.get("high_th_m", 0.0)),
            new_track_th_m=float(cfg.get("new_track_th_m", 0.0)),
            match_th_first_m=float(cfg.get("match_th_first_m", 0.0)),
            obb_theta_damping=float(cfg.get("obb_theta_damping", 0.8)),
            marginal_timeout=int(cfg.get("marginal_timeout", 0)),
            central_timeout=int(cfg.get("central_timeout", 0)),
            frame_width=int(cfg.get("frame_width", 0) or 0),
            frame_height=int(cfg.get("frame_height", 0) or 0),
            horizontal_margin=int(cfg.get("horizontal_margin", 0) or 0),
            vertical_margin=int(cfg.get("vertical_margin", 0) or 0),
            frame_rate=int(cfg.get("frame_rate", 30)),
            max_obs=int(cfg.get("max_obs", 50)),
        )
        handle = self._library.boxmot_sfsort_create(ctypes.byref(c_cfg))
        if not handle:
            raise RuntimeError(self._last_error())
        return handle

    def destroy(self, handle) -> None:
        if handle:
            self._library.boxmot_sfsort_destroy(handle)

    def reset(self, handle) -> None:
        if self._library.boxmot_sfsort_reset(handle) == 0:
            raise RuntimeError(self._last_error())

    def update(self, handle, dets: np.ndarray, img: np.ndarray) -> np.ndarray:
        return _native_trackers.call_update(
            self._library.boxmot_sfsort_update,
            handle=handle,
            dets=dets,
            img=img,
            display_name=_NATIVE_DISPLAY_NAME,
            last_error=self._last_error,
        )


def _get_live_sfsort_library() -> _SFSORTLiveLibrary:
    global _LIVE_LIBRARY
    with _LIVE_LIBRARY_LOCK:
        if _LIVE_LIBRARY is None:
            _LIVE_LIBRARY = _SFSORTLiveLibrary(ensure_sfsort_cpp_library())
        return _LIVE_LIBRARY


class NativeSFSORTTracker(_native_trackers.NativeTrackerMixin):
    supports_obb = True
    tracker_name = "sfsort"
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
        library: _SFSORTLiveLibrary | None = None,
    ) -> None:
        del reid_weights, reid_preprocess
        native_library = library if library is not None else _get_live_sfsort_library()
        self._init_native_handle(library=native_library, cfg=_resolve_tracker_cfg(cfg_dict))

    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray | None = None) -> np.ndarray:
        del embs
        det_arr = self._coerce_detections_for_mode(dets)
        return self._library.update(self._handle, det_arr, img)


def create_sfsort_live_tracker(
    cfg_dict: dict[str, Any] | None = None,
    *,
    reid_weights: str | Path | None = None,
    reid_preprocess: str | None = None,
) -> NativeSFSORTTracker:
    return NativeSFSORTTracker(cfg_dict, reid_weights=reid_weights, reid_preprocess=reid_preprocess)


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
    if str(tracker_name).lower() != "sfsort":
        raise ValueError("The native cpp replay backend currently supports tracker='sfsort' only.")

    executable = ensure_sfsort_cpp_executable()
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
            "--high-th",
            str(float(cfg["high_th"])),
            "--match-th-first",
            str(float(cfg["match_th_first"])),
            "--new-track-th",
            str(float(cfg["new_track_th"])),
            "--low-th",
            str(float(cfg["low_th"])),
            "--match-th-second",
            str(float(cfg["match_th_second"])),
            "--dynamic-tuning",
            _native_trackers.bool_arg(cfg.get("dynamic_tuning", False)),
            "--cth",
            str(float(cfg.get("cth", 0.5))),
            "--high-th-m",
            str(float(cfg.get("high_th_m", 0.0))),
            "--new-track-th-m",
            str(float(cfg.get("new_track_th_m", 0.0))),
            "--match-th-first-m",
            str(float(cfg.get("match_th_first_m", 0.0))),
            "--obb-theta-damping",
            str(float(cfg.get("obb_theta_damping", 0.8))),
            "--marginal-timeout",
            str(int(cfg.get("marginal_timeout", 0))),
            "--central-timeout",
            str(int(cfg.get("central_timeout", 0))),
            "--frame-width",
            str(int(cfg.get("frame_width", 0) or 0)),
            "--frame-height",
            str(int(cfg.get("frame_height", 0) or 0)),
            "--horizontal-margin",
            str(int(cfg.get("horizontal_margin", 0) or 0)),
            "--vertical-margin",
            str(int(cfg.get("vertical_margin", 0) or 0)),
            "--frame-rate",
            str(int(cfg.get("frame_rate", 30))),
        ],
    )

    return _native_trackers.run_replay_process(
        cmd=cmd,
        seq_name=seq_name,
        display_name=_NATIVE_DISPLAY_NAME,
        progress_queue=progress_queue,
        subprocess_module=subprocess,
    )
