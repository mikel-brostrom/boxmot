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

from boxmot.engine.tuning.search_space import flatten_yaml_config
from boxmot.native import _common
from boxmot.native._common import (
    drain_native_stderr as _drain_native_stderr,
)
from boxmot.native._common import (
    dets_n_embs_root,
)
from boxmot.native._common import (
    infer_onnx_output_names as _infer_onnx_output_names,
)
from boxmot.native._common import (
    native_onnx_cache_path as _native_onnx_cache_path,
)
from boxmot.native._common import (
    parse_progress_line as _parse_progress_line,
)
from boxmot.native._common import (
    resolve_reid_model_ref as _resolve_reid_model_ref,
)
from boxmot.trackers.tracker_zoo import get_tracker_config
from boxmot.utils import logger as LOGGER
from boxmot.utils.misc import resolve_model_path  # noqa: F401  (used by tests via monkeypatch)


def _default_preprocess() -> str:
    from boxmot.reid.core.preprocessing import DEFAULT_PREPROCESS
    return DEFAULT_PREPROCESS

_BUILD_LOCK = threading.Lock()
_LIVE_LIBRARY_LOCK = threading.Lock()
_LIVE_LIBRARY = None
_PROGRESS_PREFIX = _common.PROGRESS_PREFIX
_NATIVE_DISPLAY_NAME = "BoTSORT"


_TRACKER_NAME = "botsort"


def _configure_native_reid_env() -> None:
    """Prefer the stable CPU ReID path on macOS GitHub runners."""
    if sys.platform == "darwin" and os.environ.get("GITHUB_ACTIONS", "").lower() == "true":
        os.environ.setdefault("BOXMOT_REID_DEVICE", "cpu")


def _is_ci_macos_runner() -> bool:
    return sys.platform == "darwin" and os.environ.get("GITHUB_ACTIONS", "").lower() == "true"


def _resolve_tracker_cfg(cfg_dict: dict[str, Any] | None) -> dict[str, Any]:
    with open(get_tracker_config("botsort"), "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    raw = flatten_yaml_config(raw)
    resolved = {name: spec["default"] for name, spec in raw.items()}
    if cfg_dict is not None:
        resolved.update(cfg_dict)

    resolved.setdefault("fuse_first_associate", False)
    resolved.setdefault("with_reid", True)
    resolved.setdefault("frame_rate", 30)
    resolved.setdefault("use_cmc", True)
    return resolved


def _export_reid_to_onnx(weights: Path) -> Path:
    return _common.export_reid_to_onnx(weights, display_name=_NATIVE_DISPLAY_NAME)


def _ensure_native_reid_model_path(reid_weights: str | Path | None) -> Path | None:
    return _common.ensure_native_reid_model_path(
        reid_weights,
        display_name=_NATIVE_DISPLAY_NAME,
        # Forward the (possibly monkeypatched) module-level wrapper so tests
        # that patch ``_export_reid_to_onnx`` still take effect.
        exporter=lambda weights: _export_reid_to_onnx(weights),
        resolver=lambda value: _resolve_reid_model_ref(value),
    )


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


def ensure_botsort_cpp_executable(force_rebuild: bool = False) -> Path:
    return _build_target(
        target="botsort_replay",
        candidates=_common.candidate_executables(_TRACKER_NAME),
        force_rebuild=force_rebuild,
        not_found_message="Native BoTSORT build succeeded but the botsort_replay executable was not found.",
    )


def ensure_botsort_cpp_library(force_rebuild: bool = False) -> Path:
    return _build_target(
        target="botsort_capi",
        candidates=_common.candidate_libraries(_TRACKER_NAME),
        force_rebuild=force_rebuild,
        not_found_message="Native BoTSORT build succeeded but the botsort_capi shared library was not found.",
    )


class _BotSortCConfig(ctypes.Structure):
    _fields_ = [
        ("track_high_thresh", ctypes.c_float),
        ("track_low_thresh", ctypes.c_float),
        ("new_track_thresh", ctypes.c_float),
        ("track_buffer", ctypes.c_int),
        ("match_thresh", ctypes.c_float),
        ("proximity_thresh", ctypes.c_float),
        ("appearance_thresh", ctypes.c_float),
        ("cmc_method", ctypes.c_char_p),
        ("frame_rate", ctypes.c_int),
        ("fuse_first_associate", ctypes.c_int),
        ("with_reid", ctypes.c_int),
        ("max_obs", ctypes.c_int),
        ("reid_model_path", ctypes.c_char_p),
        ("reid_preprocess", ctypes.c_char_p),
    ]


class _BotSortLiveLibrary:
    def __init__(self, library_path: Path) -> None:
        self.library_path = Path(library_path)
        # Homebrew OpenCV can pull in OpenBLAS/libomp while PyTorch already ships libomp.
        # Allow the native tracker library to coexist in-process on macOS.
        if sys.platform == "darwin":
            os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        _configure_native_reid_env()
        self._library = ctypes.CDLL(str(self.library_path))
        self._configure_functions()

    def _configure_functions(self) -> None:
        self._library.boxmot_botsort_create.argtypes = [ctypes.POINTER(_BotSortCConfig)]
        self._library.boxmot_botsort_create.restype = ctypes.c_void_p
        self._library.boxmot_botsort_destroy.argtypes = [ctypes.c_void_p]
        self._library.boxmot_botsort_destroy.restype = None
        self._library.boxmot_botsort_reset.argtypes = [ctypes.c_void_p]
        self._library.boxmot_botsort_reset.restype = ctypes.c_int
        self._library.boxmot_botsort_update.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
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
        self._library.boxmot_botsort_update.restype = ctypes.c_int
        self._library.boxmot_botsort_last_reid_time_ms.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
        self._library.boxmot_botsort_last_reid_time_ms.restype = ctypes.c_int
        for sym in (
            "boxmot_botsort_last_reid_preprocess_time_ms",
            "boxmot_botsort_last_reid_process_time_ms",
            "boxmot_botsort_last_reid_postprocess_time_ms",
        ):
            fn = getattr(self._library, sym, None)
            if fn is not None:
                fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
                fn.restype = ctypes.c_int
        self._library.boxmot_botsort_last_error.argtypes = []
        self._library.boxmot_botsort_last_error.restype = ctypes.c_char_p

    def _last_error(self) -> str:
        raw = self._library.boxmot_botsort_last_error()
        if raw is None:
            return "Unknown native BoTSORT error."
        return raw.decode("utf-8", errors="replace") or "Unknown native BoTSORT error."

    def create(self, cfg: dict[str, Any]):
        cmc_method = str(cfg.get("cmc_method", "ecc")) if bool(cfg.get("use_cmc", True)) else "none"
        c_cfg = _BotSortCConfig(
            track_high_thresh=float(cfg["track_high_thresh"]),
            track_low_thresh=float(cfg["track_low_thresh"]),
            new_track_thresh=float(cfg["new_track_thresh"]),
            track_buffer=int(cfg["track_buffer"]),
            match_thresh=float(cfg["match_thresh"]),
            proximity_thresh=float(cfg["proximity_thresh"]),
            appearance_thresh=float(cfg["appearance_thresh"]),
            cmc_method=cmc_method.encode("utf-8"),
            frame_rate=int(cfg.get("frame_rate", 30)),
            fuse_first_associate=int(bool(cfg.get("fuse_first_associate", False))),
            with_reid=int(bool(cfg.get("with_reid", True))),
            max_obs=int(cfg.get("max_obs", 50)),
            reid_model_path=str(cfg.get("reid_model_path", "")).encode("utf-8"),
            reid_preprocess=str(cfg.get("reid_preprocess") or _default_preprocess()).encode("utf-8"),
        )
        handle = self._library.boxmot_botsort_create(ctypes.byref(c_cfg))
        if not handle:
            raise RuntimeError(self._last_error())
        return handle

    def destroy(self, handle) -> None:
        if handle:
            self._library.boxmot_botsort_destroy(handle)

    def reset(self, handle) -> None:
        if self._library.boxmot_botsort_reset(handle) == 0:
            raise RuntimeError(self._last_error())

    def update(self, handle, dets: np.ndarray, img: np.ndarray, embs: np.ndarray | None = None) -> np.ndarray:
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
                "Native BoTSORT live tracking supports AABB detections with 6 columns or OBB detections with 7 columns."
            )
        det_arr = np.ascontiguousarray(det_arr, dtype=np.float32)

        emb_arr = None
        if embs is not None:
            emb_arr = np.asarray(embs, dtype=np.float32)
            if emb_arr.size == 0:
                emb_arr = np.empty((0, 0), dtype=np.float32)
            elif emb_arr.ndim == 1:
                emb_arr = emb_arr.reshape(1, -1)
            if emb_arr.ndim != 2:
                raise ValueError("Embeddings must be a 2D array.")
            if emb_arr.shape[0] != det_arr.shape[0]:
                raise ValueError("Detections and embeddings must have the same number of rows.")
            emb_arr = np.ascontiguousarray(emb_arr, dtype=np.float32)

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
        ok = self._library.boxmot_botsort_update(
            handle,
            None if det_arr.size == 0 else ctypes.c_void_p(det_arr.ctypes.data),
            int(det_arr.shape[0]),
            int(det_arr.shape[1]),
            None if emb_arr is None or emb_arr.size == 0 else ctypes.c_void_p(emb_arr.ctypes.data),
            0 if emb_arr is None else int(emb_arr.shape[0]),
            0 if emb_arr is None else int(emb_arr.shape[1]),
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

    def get_last_reid_time_ms(self, handle) -> float:
        out_value = ctypes.c_double(0.0)
        ok = self._library.boxmot_botsort_last_reid_time_ms(handle, ctypes.byref(out_value))
        if ok == 0:
            raise RuntimeError(self._last_error())
        return float(out_value.value)

    def _get_phase_time_ms(self, handle, phase: str) -> float:
        sym = f"boxmot_botsort_last_reid_{phase}_time_ms"
        fn = getattr(self._library, sym, None)
        if fn is None:
            return 0.0
        out_value = ctypes.c_double(0.0)
        ok = fn(handle, ctypes.byref(out_value))
        if ok == 0:
            raise RuntimeError(self._last_error())
        return float(out_value.value)

    def get_last_reid_preprocess_time_ms(self, handle) -> float:
        return self._get_phase_time_ms(handle, "preprocess")

    def get_last_reid_process_time_ms(self, handle) -> float:
        return self._get_phase_time_ms(handle, "process")

    def get_last_reid_postprocess_time_ms(self, handle) -> float:
        return self._get_phase_time_ms(handle, "postprocess")


def _get_live_botsort_library() -> _BotSortLiveLibrary:
    global _LIVE_LIBRARY
    with _LIVE_LIBRARY_LOCK:
        if _LIVE_LIBRARY is None:
            _LIVE_LIBRARY = _BotSortLiveLibrary(ensure_botsort_cpp_library())
        return _LIVE_LIBRARY


class NativeBotSortTracker:
    supports_obb = True
    tracker_name = "botsort"
    tracker_backend = "cpp"

    def __init__(
        self,
        cfg_dict: dict[str, Any] | None = None,
        *,
        reid_weights: str | Path | None = None,
        reid_preprocess: str | None = None,
        library: _BotSortLiveLibrary | None = None,
    ) -> None:
        self.cfg = _resolve_tracker_cfg(cfg_dict)
        native_reid_path = None
        if _is_ci_macos_runner() and reid_weights is not None:
            resolved_ref = _resolve_reid_model_ref(reid_weights)
            if resolved_ref is not None and resolved_ref.suffix.lower() == ".pt":
                onnx_candidate = _native_onnx_cache_path(resolved_ref)
                if not onnx_candidate.exists():
                    LOGGER.warning(
                        "Native BoTSORT ReID is disabled on macOS CI when only '.pt' weights "
                        "are available (ONNX export can crash on this runner)."
                    )
                    self.cfg["with_reid"] = False
            if bool(self.cfg.get("with_reid", True)):
                native_reid_path = _ensure_native_reid_model_path(reid_weights)
        else:
            native_reid_path = _ensure_native_reid_model_path(reid_weights)
        self.reid_model_path = str(native_reid_path) if native_reid_path is not None else ""
        self.reid_preprocess = str(reid_preprocess or _default_preprocess())
        self.cfg["reid_model_path"] = self.reid_model_path
        self.cfg["reid_preprocess"] = self.reid_preprocess
        self.with_reid = bool(self.cfg.get("with_reid", True))
        self.provides_reid = bool(self.reid_model_path and Path(self.reid_model_path).suffix.lower() == ".onnx")
        self._library = library if library is not None else _get_live_botsort_library()
        self._handle = self._library.create(self.cfg)
        self.last_reid_time_ms = 0.0
        self.last_reid_preprocess_time_ms = 0.0
        self.last_reid_process_time_ms = 0.0
        self.last_reid_postprocess_time_ms = 0.0
        self._det_cols: int | None = None

    def reset(self) -> None:
        self._library.reset(self._handle)
        self.last_reid_time_ms = 0.0
        self.last_reid_preprocess_time_ms = 0.0
        self.last_reid_process_time_ms = 0.0
        self.last_reid_postprocess_time_ms = 0.0
        self._det_cols = None

    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray | None = None) -> np.ndarray:
        det_arr = np.asarray(dets) if dets is not None else np.empty((0, 0), dtype=np.float32)
        if det_arr.size and det_arr.ndim == 2 and det_arr.shape[1] in {6, 7}:
            if self._det_cols is None:
                self._det_cols = int(det_arr.shape[1])
            elif int(det_arr.shape[1]) != self._det_cols:
                raise ValueError("Native BoTSORT tracker cannot switch between AABB and OBB inputs after initialization.")
        elif self._det_cols is not None and det_arr.size == 0:
            det_arr = np.empty((0, self._det_cols), dtype=np.float32)

        tracks = self._library.update(self._handle, det_arr, img, embs)
        getter = getattr(self._library, "get_last_reid_time_ms", None)
        self.last_reid_time_ms = float(getter(self._handle)) if callable(getter) else 0.0
        for phase in ("preprocess", "process", "postprocess"):
            phase_getter = getattr(self._library, f"get_last_reid_{phase}_time_ms", None)
            value = float(phase_getter(self._handle)) if callable(phase_getter) else 0.0
            setattr(self, f"last_reid_{phase}_time_ms", value)
        return tracks

    def get_last_reid_time_ms(self) -> float:
        return float(getattr(self, "last_reid_time_ms", 0.0))

    def get_last_reid_preprocess_time_ms(self) -> float:
        return float(getattr(self, "last_reid_preprocess_time_ms", 0.0))

    def get_last_reid_process_time_ms(self) -> float:
        return float(getattr(self, "last_reid_process_time_ms", 0.0))

    def get_last_reid_postprocess_time_ms(self) -> float:
        return float(getattr(self, "last_reid_postprocess_time_ms", 0.0))

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


def create_botsort_live_tracker(
    cfg_dict: dict[str, Any] | None = None,
    *,
    reid_weights: str | Path | None = None,
    reid_preprocess: str | None = None,
) -> NativeBotSortTracker:
    return NativeBotSortTracker(cfg_dict, reid_weights=reid_weights, reid_preprocess=reid_preprocess)


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
    if str(tracker_name).lower() != "botsort":
        raise ValueError("The native cpp replay backend currently supports tracker='botsort' only.")

    executable = ensure_botsort_cpp_executable()
    cfg = _resolve_tracker_cfg(cfg_dict)

    detector_key = Path(detector_name).stem if Path(detector_name).suffix else str(detector_name)
    from boxmot.data.cache import reid_cache_key as _reid_cache_key
    reid_key = _reid_cache_key(reid_name, tracker_backend="cpp")
    reid_model_path = _ensure_native_reid_model_path(reid_name)

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
        "--reid-name",
        reid_key,
        "--reid-model",
        "" if reid_model_path is None else str(reid_model_path),
        "--reid-preprocess",
        str(preprocess_name or "resize"),
        "--sequence",
        seq_name,
        "--output",
        str(output_path),
        "--conf-threshold",
        str(float(conf_threshold)),
        "--target-fps",
        str(int(target_fps or 0)),
        "--track-high-thresh",
        str(float(cfg["track_high_thresh"])),
        "--track-low-thresh",
        str(float(cfg["track_low_thresh"])),
        "--new-track-thresh",
        str(float(cfg["new_track_thresh"])),
        "--track-buffer",
        str(int(cfg["track_buffer"])),
        "--match-thresh",
        str(float(cfg["match_thresh"])),
        "--proximity-thresh",
        str(float(cfg["proximity_thresh"])),
        "--appearance-thresh",
        str(float(cfg["appearance_thresh"])),
        "--cmc-method",
        str(cfg.get("cmc_method", "ecc")) if bool(cfg.get("use_cmc", True)) else "none",
        "--frame-rate",
        str(int(cfg.get("frame_rate", 30))),
        "--fuse-first-associate",
        "1" if bool(cfg.get("fuse_first_associate", False)) else "0",
        "--with-reid",
        "1" if bool(cfg.get("with_reid", True)) else "0",
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
            "Native BoTSORT replay failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"{stderr_text or stdout_text.strip()}"
        )

    summary = _parse_summary(stdout_text)
    kept_frame_ids = [int(value) for value in summary.get("kept_frame_ids", [])]
    return seq_name, kept_frame_ids, {
        "track_time_ms": float(summary.get("track_time_ms", 0.0)),
        "num_frames": int(summary.get("num_frames", len(kept_frame_ids))),
    }
