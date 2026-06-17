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
from boxmot.native._common import (
    drain_native_stderr as _drain_native_stderr,
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
from boxmot.utils.misc import resolve_model_path  # noqa: F401  (used by tests via monkeypatch)


def _default_preprocess() -> str:
    from boxmot.reid.core.preprocessing import DEFAULT_PREPROCESS
    return DEFAULT_PREPROCESS


_BUILD_LOCK = threading.Lock()
_LIVE_LIBRARY_LOCK = threading.Lock()
_LIVE_LIBRARY = None
_PROGRESS_PREFIX = _common.PROGRESS_PREFIX
_NATIVE_DISPLAY_NAME = "OccluBoost"


_TRACKER_NAME = "occluboost"


def _default_native_reid_device() -> str:
    """Prefer the stable CPU ReID path on macOS GitHub runners."""
    if sys.platform == "darwin" and os.environ.get("GITHUB_ACTIONS", "").lower() == "true":
        return "cpu"
    return "auto"


def _resolve_tracker_cfg(cfg_dict: dict[str, Any] | None) -> dict[str, Any]:
    with open(get_tracker_config("occluboost"), "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    resolved = {name: spec["default"] for name, spec in raw.items()}
    if cfg_dict is not None:
        resolved.update(cfg_dict)
    resolved.setdefault("with_reid", True)
    resolved.setdefault("cmc_method", "ecc")
    resolved.setdefault("max_obs", 50)
    resolved.setdefault("reid_device", _default_native_reid_device())
    return resolved


def _export_reid_to_onnx(weights: Path) -> Path:
    return _common.export_reid_to_onnx(weights, display_name=_NATIVE_DISPLAY_NAME)


def _ensure_native_reid_model_path(reid_weights: str | Path | None) -> Path | None:
    return _common.ensure_native_reid_model_path(
        reid_weights,
        display_name=_NATIVE_DISPLAY_NAME,
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


def ensure_occluboost_cpp_executable(force_rebuild: bool = False) -> Path:
    return _build_target(
        target="occluboost_replay",
        candidates=_common.candidate_executables(_TRACKER_NAME),
        force_rebuild=force_rebuild,
        not_found_message="Native OccluBoost build succeeded but the occluboost_replay executable was not found.",
    )


def ensure_occluboost_cpp_library(force_rebuild: bool = False) -> Path:
    return _build_target(
        target="occluboost_capi",
        candidates=_common.candidate_libraries(_TRACKER_NAME),
        force_rebuild=force_rebuild,
        not_found_message="Native OccluBoost build succeeded but the occluboost_capi shared library was not found.",
    )


class _OccluBoostCConfig(ctypes.Structure):
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
        ("with_reid", ctypes.c_int),
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
        ("reid_model_path", ctypes.c_char_p),
        ("reid_preprocess", ctypes.c_char_p),
        ("reid_device", ctypes.c_char_p),
    ]


def _build_c_config(cfg: dict[str, Any]) -> _OccluBoostCConfig:
    return _OccluBoostCConfig(
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
        with_reid=int(bool(cfg.get("with_reid", True))),
        cmc_method=str(cfg.get("cmc_method", "ecc")).encode("utf-8"),
        max_obs=int(cfg.get("max_obs", 50)),
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
        lambda_emb_multiplier=float(cfg.get("lambda_emb_multiplier", 1.5)),
        reid_model_path=str(cfg.get("reid_model_path", "")).encode("utf-8"),
        reid_preprocess=str(cfg.get("reid_preprocess") or _default_preprocess()).encode("utf-8"),
        reid_device=str(cfg.get("reid_device", "auto")).encode("utf-8"),
    )


class _OccluBoostLiveLibrary:
    def __init__(self, library_path: Path) -> None:
        self.library_path = Path(library_path)
        if sys.platform == "darwin":
            os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
            os.environ.setdefault("BOXMOT_REID_DEVICE", _default_native_reid_device())
        self._library = ctypes.CDLL(str(self.library_path))
        self._configure_functions()

    def _configure_functions(self) -> None:
        self._library.boxmot_occluboost_create.argtypes = [ctypes.POINTER(_OccluBoostCConfig)]
        self._library.boxmot_occluboost_create.restype = ctypes.c_void_p
        self._library.boxmot_occluboost_destroy.argtypes = [ctypes.c_void_p]
        self._library.boxmot_occluboost_destroy.restype = None
        self._library.boxmot_occluboost_reset.argtypes = [ctypes.c_void_p]
        self._library.boxmot_occluboost_reset.restype = ctypes.c_int
        self._library.boxmot_occluboost_update.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ]
        self._library.boxmot_occluboost_update.restype = ctypes.c_int
        self._library.boxmot_occluboost_last_reid_time_ms.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
        self._library.boxmot_occluboost_last_reid_time_ms.restype = ctypes.c_int
        for sym in (
            "boxmot_occluboost_last_reid_preprocess_time_ms",
            "boxmot_occluboost_last_reid_process_time_ms",
            "boxmot_occluboost_last_reid_postprocess_time_ms",
        ):
            fn = getattr(self._library, sym, None)
            if fn is not None:
                fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
                fn.restype = ctypes.c_int
        self._library.boxmot_occluboost_last_error.argtypes = []
        self._library.boxmot_occluboost_last_error.restype = ctypes.c_char_p

    def _last_error(self) -> str:
        raw = self._library.boxmot_occluboost_last_error()
        if raw is None:
            return "Unknown native OccluBoost error."
        return raw.decode("utf-8", errors="replace") or "Unknown native OccluBoost error."

    def create(self, cfg: dict[str, Any]):
        c_cfg = _build_c_config(cfg)
        handle = self._library.boxmot_occluboost_create(ctypes.byref(c_cfg))
        if not handle:
            raise RuntimeError(self._last_error())
        return handle

    def destroy(self, handle) -> None:
        if handle:
            self._library.boxmot_occluboost_destroy(handle)

    def reset(self, handle) -> None:
        if self._library.boxmot_occluboost_reset(handle) == 0:
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
                "Native OccluBoost live tracking supports AABB detections with 6 columns or OBB detections with 7 columns."
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
        ok = self._library.boxmot_occluboost_update(
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
        ok = self._library.boxmot_occluboost_last_reid_time_ms(handle, ctypes.byref(out_value))
        if ok == 0:
            raise RuntimeError(self._last_error())
        return float(out_value.value)

    def _get_phase_time_ms(self, handle, phase: str) -> float:
        sym = f"boxmot_occluboost_last_reid_{phase}_time_ms"
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


def _get_live_occluboost_library() -> _OccluBoostLiveLibrary:
    global _LIVE_LIBRARY
    with _LIVE_LIBRARY_LOCK:
        if _LIVE_LIBRARY is None:
            _LIVE_LIBRARY = _OccluBoostLiveLibrary(ensure_occluboost_cpp_library())
        return _LIVE_LIBRARY


class NativeOccluBoostTracker:
    supports_obb = True
    tracker_name = "occluboost"
    tracker_backend = "cpp"

    def __init__(
        self,
        cfg_dict: dict[str, Any] | None = None,
        *,
        reid_weights: str | Path | None = None,
        reid_preprocess: str | None = None,
        reid_device: str | None = None,
        library: _OccluBoostLiveLibrary | None = None,
    ) -> None:
        self.cfg = _resolve_tracker_cfg(cfg_dict)
        native_reid_path = _ensure_native_reid_model_path(reid_weights)
        self.reid_model_path = str(native_reid_path) if native_reid_path is not None else ""
        self.reid_preprocess = str(reid_preprocess or _default_preprocess())
        self.cfg["reid_model_path"] = self.reid_model_path
        self.cfg["reid_preprocess"] = self.reid_preprocess
        self.cfg["reid_device"] = str(reid_device or self.cfg.get("reid_device") or _default_native_reid_device())
        self.with_reid = bool(self.cfg.get("with_reid", True))
        self.provides_reid = bool(self.reid_model_path and Path(self.reid_model_path).suffix.lower() == ".onnx")
        self._library = library if library is not None else _get_live_occluboost_library()
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
                raise ValueError(
                    "Native OccluBoost tracker cannot switch between AABB and OBB inputs after initialization."
                )
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


def create_occluboost_live_tracker(
    cfg_dict: dict[str, Any] | None = None,
    *,
    reid_weights: str | Path | None = None,
    reid_preprocess: str | None = None,
    reid_device: str | None = None,
) -> NativeOccluBoostTracker:
    return NativeOccluBoostTracker(cfg_dict, reid_weights=reid_weights, reid_preprocess=reid_preprocess, reid_device=reid_device)


def _parse_summary(stdout: str) -> dict[str, Any]:
    return _common.parse_summary(stdout, display_name=_NATIVE_DISPLAY_NAME)


def _bool_arg(value: Any) -> str:
    return "1" if bool(value) else "0"


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
    if str(tracker_name).lower() != "occluboost":
        raise ValueError("The native cpp replay backend currently supports tracker='occluboost' only.")

    executable = ensure_occluboost_cpp_executable()
    cfg = _resolve_tracker_cfg(cfg_dict)

    detector_key = Path(detector_name).stem if Path(detector_name).suffix else str(detector_name)
    from boxmot.data.cache import reid_cache_key as _reid_cache_key
    reid_key = _reid_cache_key(reid_name, tracker_backend="cpp")

    det_emb_root = dets_n_embs_root(project_root, dataset_name, split=split)

    # Skip the (potentially expensive) ONNX export + model load when a complete
    # embedding cache already exists for this sequence; the C++ tracker will read
    # embeddings straight from the .npy and never invoke ReID.
    preprocess_key = str(preprocess_name or "resize")
    cached_emb = cached_embedding_path(
        project_root,
        detector_name,
        reid_name,
        seq_name,
        dataset_name=dataset_name,
        preprocess_name=preprocess_key,
        tracker_backend="cpp",
    )
    if cached_emb.exists():
        reid_model_path = None
    else:
        reid_model_path = _ensure_native_reid_model_path(reid_name)

    output_path = Path(exp_folder) / f"{seq_name}.txt"
    cmd = [
        str(executable),
        "--mot-root", str(mot_root),
        "--det-emb-root", str(det_emb_root),
        "--detector-name", detector_key,
        "--reid-name", reid_key,
        "--reid-model", "" if reid_model_path is None else str(reid_model_path),
        "--reid-preprocess", preprocess_key,
        "--sequence", seq_name,
        "--output", str(output_path),
        "--conf-threshold", str(float(conf_threshold)),
        "--target-fps", str(int(target_fps or 0)),
        "--max-age", str(int(cfg["max_age"])),
        "--min-hits", str(int(cfg["min_hits"])),
        "--det-thresh", str(float(cfg["det_thresh"])),
        "--iou-threshold", str(float(cfg["iou_threshold"])),
        "--min-box-area", str(int(cfg["min_box_area"])),
        "--aspect-ratio-thresh", str(float(cfg["aspect_ratio_thresh"])),
        "--lambda-iou", str(float(cfg["lambda_iou"])),
        "--lambda-mhd", str(float(cfg["lambda_mhd"])),
        "--lambda-shape", str(float(cfg["lambda_shape"])),
        "--use-dlo-boost", _bool_arg(cfg["use_dlo_boost"]),
        "--use-duo-boost", _bool_arg(cfg["use_duo_boost"]),
        "--dlo-boost-coef", str(float(cfg["dlo_boost_coef"])),
        "--s-sim-corr", _bool_arg(cfg["s_sim_corr"]),
        "--use-rich-s", _bool_arg(cfg["use_rich_s"]),
        "--use-sb", _bool_arg(cfg["use_sb"]),
        "--use-vt", _bool_arg(cfg["use_vt"]),
        "--with-reid", _bool_arg(cfg.get("with_reid", True)),
        "--cmc-method", str(cfg.get("cmc_method", "ecc")),
        "--max-obs", str(int(cfg.get("max_obs", 50))),
        "--recovery-appearance-thresh", str(float(cfg["recovery_appearance_thresh"])),
        "--recovery-iou-thresh", str(float(cfg["recovery_iou_thresh"])),
        "--recovery-max-age", str(int(cfg["recovery_max_age"])),
        "--feat-alpha", str(float(cfg["feat_alpha"])),
        "--track-low-thresh", str(float(cfg["track_low_thresh"])),
        "--second-iou-thresh", str(float(cfg["second_iou_thresh"])),
        "--second-appearance-thresh", str(float(cfg["second_appearance_thresh"])),
        "--second-pass-max-age", str(int(cfg["second_pass_max_age"])),
        "--second-pass-min-hits", str(int(cfg["second_pass_min_hits"])),
        "--use-second-pass", _bool_arg(cfg["use_second_pass"]),
        "--new-track-thresh", str(float(cfg["new_track_thresh"])),
        "--confirm-hits", str(int(cfg["confirm_hits"])),
        "--instant-confirm-thresh", str(float(cfg["instant_confirm_thresh"])),
        "--tentative-max-age", str(int(cfg["tentative_max_age"])),
        "--duplicate-iou-thresh", str(float(cfg["duplicate_iou_thresh"])),
        "--ams-enabled", _bool_arg(cfg["ams_enabled"]),
        "--ams-alpha0", str(float(cfg["ams_alpha0"])),
        "--ams-threshold", str(float(cfg["ams_threshold"])),
        "--ams-buffer-size", str(int(cfg["ams_buffer_size"])),
        "--ams-shrink-ratio", str(float(cfg["ams_shrink_ratio"])),
        "--lambda-emb-multiplier", str(float(cfg.get("lambda_emb_multiplier", 1.5))),
    ]

    # When the eval pipeline runs N sequences in parallel via a process pool,
    # each occluboost_replay subprocess can otherwise spin up its own OpenCV /
    # OpenMP / Apple GCD thread pool, leading to N*cores threads contending for
    # the same cores. The result on macOS is that progress crawls or visibly
    # freezes. Cap the per-subprocess thread count via env vars BEFORE the
    # process loads OpenCV (cv::setNumThreads cannot affect the Apple GCD
    # backend that OpenCV uses for parallel_for_).
    subprocess_env = dict(os.environ)
    if not subprocess_env.get("BOXMOT_NATIVE_CV_THREADS"):
        subprocess_env.setdefault("OMP_NUM_THREADS", "1")
        subprocess_env.setdefault("OPENBLAS_NUM_THREADS", "1")
        subprocess_env.setdefault("MKL_NUM_THREADS", "1")
        subprocess_env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        subprocess_env.setdefault("NUMEXPR_NUM_THREADS", "1")
        subprocess_env.setdefault("OPENCV_FOR_THREADS_NUM", "1")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=subprocess_env,
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
            "Native OccluBoost replay failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"{stderr_text or stdout_text.strip()}"
        )

    summary = _parse_summary(stdout_text)
    kept_frame_ids = [int(value) for value in summary.get("kept_frame_ids", [])]
    return seq_name, kept_frame_ids, {
        "track_time_ms": float(summary.get("track_time_ms", 0.0)),
        "num_frames": int(summary.get("num_frames", len(kept_frame_ids))),
    }
