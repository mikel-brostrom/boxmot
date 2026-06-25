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
_NATIVE_DISPLAY_NAME = "ByteTrack"


_TRACKER_NAME = "bytetrack"


def _resolve_tracker_cfg(cfg_dict: dict[str, Any] | None) -> dict[str, Any]:
    resolved = _native_trackers.load_tracker_cfg("bytetrack", cfg_dict)
    resolved.setdefault("frame_rate", 30)
    resolved.setdefault("max_obs", 50)
    return resolved


def ensure_bytetrack_cpp_executable(force_rebuild: bool = False) -> Path:
    return _native_trackers.ensure_tracker_executable(
        tracker_name=_TRACKER_NAME,
        display_name=_NATIVE_DISPLAY_NAME,
        build_lock=_BUILD_LOCK,
        force_rebuild=force_rebuild,
    )


def ensure_bytetrack_cpp_library(force_rebuild: bool = False) -> Path:
    return _native_trackers.ensure_tracker_library(
        tracker_name=_TRACKER_NAME,
        display_name=_NATIVE_DISPLAY_NAME,
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
    ]


class _ByteTrackLiveLibrary:
    def __init__(self, library_path: Path) -> None:
        self.library_path = Path(library_path)
        if sys.platform == "darwin":
            os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        self._library = ctypes.CDLL(str(self.library_path))
        self._configure_functions()

    def _configure_functions(self) -> None:
        self._library.boxmot_bytetrack_create.argtypes = [ctypes.POINTER(_ByteTrackCConfig)]
        self._library.boxmot_bytetrack_create.restype = ctypes.c_void_p
        self._library.boxmot_bytetrack_destroy.argtypes = [ctypes.c_void_p]
        self._library.boxmot_bytetrack_destroy.restype = None
        self._library.boxmot_bytetrack_reset.argtypes = [ctypes.c_void_p]
        self._library.boxmot_bytetrack_reset.restype = ctypes.c_int
        self._library.boxmot_bytetrack_update.argtypes = _native_trackers.LIVE_UPDATE_ARGTYPES
        self._library.boxmot_bytetrack_update.restype = ctypes.c_int
        self._library.boxmot_bytetrack_last_error.argtypes = []
        self._library.boxmot_bytetrack_last_error.restype = ctypes.c_char_p

    def _last_error(self) -> str:
        raw = self._library.boxmot_bytetrack_last_error()
        if raw is None:
            return "Unknown native ByteTrack error."
        return raw.decode("utf-8", errors="replace") or "Unknown native ByteTrack error."

    def create(self, cfg: dict[str, Any]):
        c_cfg = _ByteTrackCConfig(
            min_conf=float(cfg["min_conf"]),
            track_thresh=float(cfg["track_thresh"]),
            match_thresh=float(cfg["match_thresh"]),
            track_buffer=int(cfg["track_buffer"]),
            frame_rate=int(cfg.get("frame_rate", 30)),
            max_obs=int(cfg.get("max_obs", 50)),
        )
        handle = self._library.boxmot_bytetrack_create(ctypes.byref(c_cfg))
        if not handle:
            raise RuntimeError(self._last_error())
        return handle

    def destroy(self, handle) -> None:
        if handle:
            self._library.boxmot_bytetrack_destroy(handle)

    def reset(self, handle) -> None:
        if self._library.boxmot_bytetrack_reset(handle) == 0:
            raise RuntimeError(self._last_error())

    def update(self, handle, dets: np.ndarray, img: np.ndarray) -> np.ndarray:
        return _native_trackers.call_update(
            self._library.boxmot_bytetrack_update,
            handle=handle,
            dets=dets,
            img=img,
            display_name=_NATIVE_DISPLAY_NAME,
            last_error=self._last_error,
        )


def _get_live_bytetrack_library() -> _ByteTrackLiveLibrary:
    global _LIVE_LIBRARY
    with _LIVE_LIBRARY_LOCK:
        if _LIVE_LIBRARY is None:
            _LIVE_LIBRARY = _ByteTrackLiveLibrary(ensure_bytetrack_cpp_library())
        return _LIVE_LIBRARY


class NativeByteTrackTracker(_native_trackers.NativeTrackerMixin):
    supports_obb = True
    tracker_name = "bytetrack"
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
        library: _ByteTrackLiveLibrary | None = None,
    ) -> None:
        del reid_weights, reid_preprocess
        native_library = library if library is not None else _get_live_bytetrack_library()
        self._init_native_handle(library=native_library, cfg=_resolve_tracker_cfg(cfg_dict))

    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray | None = None) -> np.ndarray:
        del embs
        det_arr = self._coerce_detections_for_mode(dets)
        return self._library.update(self._handle, det_arr, img)


def create_bytetrack_live_tracker(
    cfg_dict: dict[str, Any] | None = None,
    *,
    reid_weights: str | Path | None = None,
    reid_preprocess: str | None = None,
) -> NativeByteTrackTracker:
    return NativeByteTrackTracker(cfg_dict, reid_weights=reid_weights, reid_preprocess=reid_preprocess)


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
    if str(tracker_name).lower() != "bytetrack":
        raise ValueError("The native cpp replay backend currently supports tracker='bytetrack' only.")

    executable = ensure_bytetrack_cpp_executable()
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
            "--track-thresh",
            str(float(cfg["track_thresh"])),
            "--track-buffer",
            str(int(cfg["track_buffer"])),
            "--match-thresh",
            str(float(cfg["match_thresh"])),
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
