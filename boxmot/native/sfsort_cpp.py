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
)
from boxmot.native._common import (
    drain_native_stderr as _drain_native_stderr,
)
from boxmot.native._common import (
    parse_progress_line as _parse_progress_line,
)
from boxmot.trackers.tracker_zoo import get_tracker_config

_BUILD_LOCK = threading.Lock()
_LIVE_LIBRARY_LOCK = threading.Lock()
_LIVE_LIBRARY = None
_PROGRESS_PREFIX = _common.PROGRESS_PREFIX
_NATIVE_DISPLAY_NAME = "SFSORT"


_TRACKER_NAME = "sfsort"


def _resolve_tracker_cfg(cfg_dict: dict[str, Any] | None) -> dict[str, Any]:
    with open(get_tracker_config("sfsort"), "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    resolved = {name: spec["default"] for name, spec in raw.items()}
    if cfg_dict is not None:
        resolved.update(cfg_dict)

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


def ensure_sfsort_cpp_executable(force_rebuild: bool = False) -> Path:
    return _build_target(
        target="sfsort_replay",
        candidates=_common.candidate_executables(_TRACKER_NAME),
        force_rebuild=force_rebuild,
        not_found_message="Native SFSORT build succeeded but the sfsort_replay executable was not found.",
    )


def ensure_sfsort_cpp_library(force_rebuild: bool = False) -> Path:
    return _build_target(
        target="sfsort_capi",
        candidates=_common.candidate_libraries(_TRACKER_NAME),
        force_rebuild=force_rebuild,
        not_found_message="Native SFSORT build succeeded but the sfsort_capi shared library was not found.",
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
        self._library.boxmot_sfsort_update.argtypes = [
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
                "Native SFSORT live tracking supports AABB detections with 6 columns or OBB detections with 7 columns."
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
        ok = self._library.boxmot_sfsort_update(
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


def _get_live_sfsort_library() -> _SFSORTLiveLibrary:
    global _LIVE_LIBRARY
    with _LIVE_LIBRARY_LOCK:
        if _LIVE_LIBRARY is None:
            _LIVE_LIBRARY = _SFSORTLiveLibrary(ensure_sfsort_cpp_library())
        return _LIVE_LIBRARY


class NativeSFSORTTracker:
    supports_obb = True
    tracker_name = "sfsort"
    tracker_backend = "cpp"
    provides_reid = False
    with_reid = False

    def __init__(
        self,
        cfg_dict: dict[str, Any] | None = None,
        *,
        reid_weights: str | Path | None = None,
        reid_preprocess: str | None = None,
        library: _SFSORTLiveLibrary | None = None,
    ) -> None:
        del reid_weights, reid_preprocess
        self.cfg = _resolve_tracker_cfg(cfg_dict)
        self._library = library if library is not None else _get_live_sfsort_library()
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
                raise ValueError("Native SFSORT tracker cannot switch between AABB and OBB inputs after initialization.")
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
    progress_queue=None,
):
    del reid_name, preprocess_name
    if str(tracker_name).lower() != "sfsort":
        raise ValueError("The native cpp replay backend currently supports tracker='sfsort' only.")

    executable = ensure_sfsort_cpp_executable()
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
        "1" if bool(cfg.get("dynamic_tuning", False)) else "0",
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
            "Native SFSORT replay failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"{stderr_text or stdout_text.strip() or 'Unknown native SFSORT failure.'}"
        )

    summary = _parse_summary(stdout_text)
    if str(summary.get("sequence")) != str(seq_name):
        raise RuntimeError(
            "Native SFSORT summary sequence mismatch: "
            f"expected {seq_name!r}, got {summary.get('sequence')!r}."
        )

    kept_ids = [int(frame_id) for frame_id in summary.get("kept_frame_ids", [])]
    timing = {
        "track_time_ms": float(summary.get("track_time_ms", 0.0)),
        "num_frames": int(summary.get("num_frames", len(kept_ids))),
    }
    return seq_name, kept_ids, timing
