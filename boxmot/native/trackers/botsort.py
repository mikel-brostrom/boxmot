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
from boxmot.native._common import (
    dets_n_embs_root,
)
from boxmot.native._common import (
    native_onnx_cache_path as _native_onnx_cache_path,
)
from boxmot.native._common import (
    resolve_reid_model_ref as _resolve_reid_model_ref,
)
from boxmot.native.trackers import _common as _native_trackers
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
    resolved = _native_trackers.load_tracker_cfg("botsort", cfg_dict, flatten=True)
    _native_trackers.resolve_association_function(resolved)
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


def ensure_botsort_cpp_executable(force_rebuild: bool = False) -> Path:
    return _native_trackers.ensure_tracker_executable(
        tracker_name=_TRACKER_NAME,
        display_name=_NATIVE_DISPLAY_NAME,
        build_lock=_BUILD_LOCK,
        force_rebuild=force_rebuild,
    )


def ensure_botsort_cpp_library(force_rebuild: bool = False) -> Path:
    return _native_trackers.ensure_tracker_library(
        tracker_name=_TRACKER_NAME,
        display_name=_NATIVE_DISPLAY_NAME,
        build_lock=_BUILD_LOCK,
        force_rebuild=force_rebuild,
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
        ("second_match_thresh", ctypes.c_float),
        ("unconfirmed_match_thresh", ctypes.c_float),
        ("unconfirmed_emb_scale", ctypes.c_float),
        ("cmc_method", ctypes.c_char_p),
        ("frame_rate", ctypes.c_int),
        ("fuse_first_associate", ctypes.c_int),
        ("with_reid", ctypes.c_int),
        ("max_obs", ctypes.c_int),
        ("reid_model_path", ctypes.c_char_p),
        ("reid_preprocess", ctypes.c_char_p),
        ("asso_func", ctypes.c_char_p),
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
        self._library.boxmot_botsort_update.argtypes = _native_trackers.LIVE_UPDATE_WITH_EMBS_ARGTYPES
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
            second_match_thresh=float(cfg["second_match_thresh"]),
            unconfirmed_match_thresh=float(cfg["unconfirmed_match_thresh"]),
            unconfirmed_emb_scale=float(cfg["unconfirmed_emb_scale"]),
            cmc_method=cmc_method.encode("utf-8"),
            frame_rate=int(cfg.get("frame_rate", 30)),
            fuse_first_associate=int(bool(cfg.get("fuse_first_associate", False))),
            with_reid=int(bool(cfg.get("with_reid", True))),
            max_obs=int(cfg.get("max_obs", 50)),
            reid_model_path=str(cfg.get("reid_model_path", "")).encode("utf-8"),
            reid_preprocess=str(cfg.get("reid_preprocess") or _default_preprocess()).encode("utf-8"),
            asso_func=str(cfg["asso_func"]).encode("utf-8"),
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

    def update(
        self,
        handle,
        dets: np.ndarray,
        img: np.ndarray | None,
        embs: np.ndarray | None = None,
    ) -> np.ndarray:
        return _native_trackers.call_update(
            self._library.boxmot_botsort_update,
            handle=handle,
            dets=dets,
            img=img,
            embs=embs,
            accepts_embeddings=True,
            display_name=_NATIVE_DISPLAY_NAME,
            last_error=self._last_error,
            requires_image=False,
        )

    def get_last_reid_time_ms(self, handle) -> float:
        return _native_trackers.get_double_result(
            self._library,
            "boxmot_botsort_last_reid_time_ms",
            handle,
            self._last_error,
        )

    def _get_phase_time_ms(self, handle, phase: str) -> float:
        return _native_trackers.get_double_result(
            self._library,
            f"boxmot_botsort_last_reid_{phase}_time_ms",
            handle,
            self._last_error,
        )

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


class NativeBotSortTracker(_native_trackers.NativeTrackerMixin):
    supports_obb = True
    supports_masks = False
    uses_img = True
    uses_embs = True
    tracker_name = "botsort"
    tracker_backend = "cpp"
    _native_display_name = _NATIVE_DISPLAY_NAME
    _tracks_reid_timing = True

    def __init__(
        self,
        cfg_dict: dict[str, Any] | None = None,
        *,
        reid_weights: str | Path | None = None,
        reid_preprocess: str | None = None,
        library: _BotSortLiveLibrary | None = None,
    ) -> None:
        cfg = _resolve_tracker_cfg(cfg_dict)
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
                    cfg["with_reid"] = False
            if bool(cfg.get("with_reid", True)):
                native_reid_path = _ensure_native_reid_model_path(reid_weights)
        elif bool(cfg.get("with_reid", True)):
            native_reid_path = _ensure_native_reid_model_path(reid_weights)
        self.reid_model_path = str(native_reid_path) if native_reid_path is not None else ""
        self.reid_preprocess = str(reid_preprocess or _default_preprocess())
        cfg["reid_model_path"] = self.reid_model_path
        cfg["reid_preprocess"] = self.reid_preprocess
        self.with_reid = bool(cfg.get("with_reid", True))
        self.uses_embs = self.with_reid
        self.provides_reid = bool(
            self.with_reid
            and self.reid_model_path
            and Path(self.reid_model_path).suffix.lower() == ".onnx"
        )
        self.use_cmc = bool(cfg.get("use_cmc", True))
        self.asso_func_name = cfg["asso_func"]
        self._needs_association_image = _native_trackers.association_requires_initial_image(cfg)
        self.uses_img = self.use_cmc or self.provides_reid or self._needs_association_image
        native_library = library if library is not None else _get_live_botsort_library()
        self._init_native_handle(library=native_library, cfg=cfg)

    def requires_image(
        self,
        dets: np.ndarray,
        embs: np.ndarray | None = None,
        masks: np.ndarray | None = None,
    ) -> bool:
        """Require pixels for active CMC or native extraction of missing embeddings."""
        del masks
        return self._needs_association_image or self.use_cmc or bool(self.provides_reid and len(dets) and embs is None)

    def update(
        self,
        dets: np.ndarray,
        img: np.ndarray | None = None,
        embs: np.ndarray | None = None,
        masks: np.ndarray | None = None,
    ) -> np.ndarray:
        del masks
        det_arr = self._coerce_detections_for_mode(dets)
        if self.requires_image(det_arr, embs=embs) and img is None:
            raise ValueError("Native BoTSORT requires img for live tracking.")
        emb_arr = _native_trackers.normalize_embeddings(embs, rows=int(det_arr.shape[0])) if self.uses_embs else None
        routed_img = img if self.requires_image(det_arr, embs=emb_arr) else None
        tracks = self._library.update(self._handle, det_arr, routed_img, emb_arr)
        self._needs_association_image = False
        self.uses_img = self.use_cmc or self.provides_reid
        self._refresh_reid_timings()
        return self._normalize_tracks_for_mode(tracks)

    def reset(self) -> None:
        super().reset()
        self._needs_association_image = self.asso_func_name == "centroid"
        self.uses_img = self.use_cmc or self.provides_reid or self._needs_association_image


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
    embedding_cache_dir: str | None = None,
    progress_queue=None,
    adaptive_kf: bool = False,
):
    if str(tracker_name).lower() != "botsort":
        raise ValueError("The native cpp replay backend currently supports tracker='botsort' only.")

    executable = ensure_botsort_cpp_executable()
    cfg = _resolve_tracker_cfg(cfg_dict)

    det_emb_root = dets_n_embs_root(project_root, dataset_name, split=split)
    reid_key, preprocess_key, cached_emb, cached_dets = _common.resolve_embedding_cache_location(
        project_root,
        detector_name,
        reid_name,
        seq_name,
        dataset_name=dataset_name,
        split=split,
        preprocess_name=preprocess_name,
        tracker_backend="cpp",
        embedding_cache_dir=embedding_cache_dir,
    )
    reid_model_path = (
        None
        if _common.embedding_cache_is_complete(cached_emb, cached_dets)
        else _ensure_native_reid_model_path(reid_name)
    )

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
            "--asso-func",
            str(cfg["asso_func"]),
            "--reid-name",
            reid_key,
            "--reid-model",
            "" if reid_model_path is None else str(reid_model_path),
            "--reid-preprocess",
            preprocess_key,
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
            "--second-match-thresh",
            str(float(cfg["second_match_thresh"])),
            "--unconfirmed-match-thresh",
            str(float(cfg["unconfirmed_match_thresh"])),
            "--unconfirmed-emb-scale",
            str(float(cfg["unconfirmed_emb_scale"])),
            "--cmc-method",
            str(cfg.get("cmc_method", "ecc")) if bool(cfg.get("use_cmc", True)) else "none",
            "--frame-rate",
            str(int(cfg.get("frame_rate", 30))),
            "--fuse-first-associate",
            _native_trackers.bool_arg(cfg.get("fuse_first_associate", False)),
            "--with-reid",
            _native_trackers.bool_arg(cfg.get("with_reid", True)),
        ],
    )

    return _native_trackers.run_replay_process(
        cmd=cmd,
        seq_name=seq_name,
        display_name=_NATIVE_DISPLAY_NAME,
        progress_queue=progress_queue,
        subprocess_module=subprocess,
    )
