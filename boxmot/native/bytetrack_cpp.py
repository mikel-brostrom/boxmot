from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from boxmot.native import _common
from boxmot.native._common import (  # noqa: F401
    dets_n_embs_root,
    drain_native_stderr as _drain_native_stderr,
    parse_progress_line as _parse_progress_line,
)
from boxmot.trackers.tracker_zoo import get_tracker_config

_BUILD_LOCK = threading.Lock()
_LIVE_LIBRARY_LOCK = threading.Lock()
_LIVE_LIBRARY = None
_PROGRESS_PREFIX = _common.PROGRESS_PREFIX
_NATIVE_DISPLAY_NAME = "ByteTrack"


_TRACKER_NAME = "bytetrack"


def _resolve_tracker_cfg(cfg_dict: dict[str, Any] | None) -> dict[str, Any]:
    with open(get_tracker_config("bytetrack"), "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    resolved = {name: spec["default"] for name, spec in raw.items()}
    if cfg_dict is not None:
        resolved.update(cfg_dict)

    resolved.setdefault("frame_rate", 30)
    resolved.setdefault("max_obs", 50)
    return resolved


def _build_target(*, target: str, candidates: list[Path], force_rebuild: bool, not_found_message: str) -> Path:
    return _common.build_native_target(
        tracker_name=_TRACKER_NAME,
        display_name=_NATIVE_DISPLAY_NAME,
        target=target,
        candidates=candidates,
        force_rebuild=force_rebuild,
        not_found_message=not_found_message,
        build_lock=_BUILD_LOCK,
    )


def ensure_bytetrack_cpp_executable(force_rebuild: bool = False) -> Path:
    return _build_target(
        target="bytetrack_replay",
        candidates=_common.candidate_executables(_TRACKER_NAME),
        force_rebuild=force_rebuild,
        not_found_message="Native ByteTrack build succeeded but the bytetrack_replay executable was not found.",
    )


def ensure_bytetrack_cpp_library(force_rebuild: bool = False) -> Path:
    return _build_target(
        target="bytetrack_capi",
        candidates=_common.candidate_libraries(_TRACKER_NAME),
        force_rebuild=force_rebuild,
        not_found_message="Native ByteTrack build succeeded but the bytetrack_capi shared library was not found.",
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
        self._library.boxmot_bytetrack_update.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
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
        det_arr = np.asarray(dets, dtype=np.float32)
        if det_arr.size == 0:
            if det_arr.ndim == 2 and det_arr.shape[1] in {6, 7}:
                det_arr = np.empty((0, det_arr.shape[1]), dtype=np.float32)
            else:
                det_arr = np.empty((0, 6), dtype=np.float32)
        elif det_arr.ndim == 1:
            det_arr = det_arr.reshape(1, -1)
        if det_arr.ndim != 2:
            raise ValueError("Detections must be a 2D array.")
        if det_arr.shape[1] not in {6, 7}:
            raise NotImplementedError(
                "Native ByteTrack live tracking supports AABB detections with 6 columns or OBB detections with 7 columns."
            )
        det_arr = np.ascontiguousarray(det_arr, dtype=np.float32)

        img_arr = np.asarray(img)
        if img_arr.dtype != np.uint8:
            img_arr = img_arr.astype(np.uint8, copy=False)
        if img_arr.ndim not in {2, 3}:
            raise ValueError("Image must be a 2D or 3D uint8 array.")
        img_arr = np.ascontiguousarray(img_arr)
        img_channels = 1 if img_arr.ndim == 2 else int(img_arr.shape[2])

        out_capacity = max(int(det_arr.shape[0]), 1)
        out_arr = np.empty((out_capacity, 9), dtype=np.float32)
        out_rows = ctypes.c_int(0)
        out_is_obb = ctypes.c_int(0)
        ok = self._library.boxmot_bytetrack_update(
            handle,
            None if det_arr.size == 0 else ctypes.c_void_p(det_arr.ctypes.data),
            int(det_arr.shape[0]),
            int(det_arr.shape[1]),
            ctypes.c_void_p(img_arr.ctypes.data),
            int(img_arr.shape[0]),
            int(img_arr.shape[1]),
            img_channels,
            ctypes.c_void_p(out_arr.ctypes.data),
            int(out_arr.shape[0]),
            int(out_arr.shape[1]),
            ctypes.byref(out_rows),
            ctypes.byref(out_is_obb),
        )
        if ok == 0:
            raise RuntimeError(self._last_error())
        rows = max(int(out_rows.value), 0)
        cols = 9 if bool(out_is_obb.value) else 8
        return out_arr[:rows, :cols].copy()


def _get_live_bytetrack_library() -> _ByteTrackLiveLibrary:
    global _LIVE_LIBRARY
    with _LIVE_LIBRARY_LOCK:
        if _LIVE_LIBRARY is None:
            _LIVE_LIBRARY = _ByteTrackLiveLibrary(ensure_bytetrack_cpp_library())
        return _LIVE_LIBRARY


class NativeByteTrackTracker:
    supports_obb = True
    tracker_name = "bytetrack"
    tracker_backend = "cpp"
    provides_reid = False
    with_reid = False

    def __init__(
        self,
        cfg_dict: dict[str, Any] | None = None,
        *,
        reid_weights: str | Path | None = None,
        reid_preprocess: str | None = None,
        library: _ByteTrackLiveLibrary | None = None,
    ) -> None:
        del reid_weights, reid_preprocess
        self.cfg = _resolve_tracker_cfg(cfg_dict)
        self._library = library if library is not None else _get_live_bytetrack_library()
        self._handle = self._library.create(self.cfg)
        self._det_cols: int | None = None

    def reset(self) -> None:
        self._library.reset(self._handle)
        self._det_cols = None

    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray | None = None) -> np.ndarray:
        del embs
        det_arr = np.asarray(dets) if dets is not None else np.empty((0, 0), dtype=np.float32)
        if det_arr.size and det_arr.ndim == 2 and det_arr.shape[1] in {6, 7}:
            if self._det_cols is None:
                self._det_cols = int(det_arr.shape[1])
            elif int(det_arr.shape[1]) != self._det_cols:
                raise ValueError("Native ByteTrack tracker cannot switch between AABB and OBB inputs after initialization.")
        elif self._det_cols is not None and det_arr.size == 0:
            det_arr = np.empty((0, self._det_cols), dtype=np.float32)
        return self._library.update(self._handle, det_arr, img)

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is None:
            return
        self._library.destroy(handle)
        self._handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


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
    progress_queue=None,
):
    del reid_name, preprocess_name
    if str(tracker_name).lower() != "bytetrack":
        raise ValueError("The native cpp replay backend currently supports tracker='bytetrack' only.")

    executable = ensure_bytetrack_cpp_executable()
    cfg = _resolve_tracker_cfg(cfg_dict)

    detector_key = Path(detector_name).stem if Path(detector_name).suffix else str(detector_name)

    det_emb_root = dets_n_embs_root(project_root, dataset_name, split=split)

    output_path = Path(exp_folder) / f"{seq_name}.txt"
    cmd = [
        str(executable),
        "--mot-root",
        str(mot_root),
        "--det-emb-root",
        str(det_emb_root),
        "--detector-name",
        detector_key,
        "--sequence",
        seq_name,
        "--output",
        str(output_path),
        "--conf-threshold",
        str(float(conf_threshold)),
        "--target-fps",
        str(int(target_fps or 0)),
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
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    stderr_lines: list[str] = []
    stderr_thread = threading.Thread(
        target=_drain_native_stderr,
        args=(process.stderr, progress_queue, stderr_lines),
        daemon=True,
    )
    stderr_thread.start()

    stdout_text = process.stdout.read() if process.stdout is not None else ""
    returncode = process.wait()
    stderr_thread.join()

    if process.stdout is not None:
        process.stdout.close()
    if process.stderr is not None:
        process.stderr.close()

    stderr_text = "\n".join(stderr_lines).strip()
    if returncode != 0:
        raise RuntimeError(
            "Native ByteTrack replay failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"{stderr_text or stdout_text.strip() or 'Unknown native ByteTrack failure.'}"
        )

    summary = _parse_summary(stdout_text)
    if str(summary.get("sequence")) != str(seq_name):
        raise RuntimeError(
            "Native ByteTrack summary sequence mismatch: "
            f"expected {seq_name!r}, got {summary.get('sequence')!r}."
        )

    kept_ids = [int(frame_id) for frame_id in summary.get("kept_frame_ids", [])]
    timing = {
        "track_time_ms": float(summary.get("track_time_ms", 0.0)),
        "num_frames": int(summary.get("num_frames", len(kept_ids))),
    }
    return seq_name, kept_ids, timing
