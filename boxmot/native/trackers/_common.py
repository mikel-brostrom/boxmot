"""Shared Python plumbing for native tracker bindings."""

from __future__ import annotations

import ctypes
import os
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from boxmot.core.box_schema import (
    AABB_SCHEMA,
    OBB_SCHEMA,
    get_box_schema_for_mode,
    schema_from_detection_columns,
)
from boxmot.native import _common as native_common
from boxmot.trackers.common.detections.layout import get_detection_layout
from boxmot.trackers.common.tracking.classes import ClassCatalog
from boxmot.trackers.config import load_tracker_defaults
from boxmot.trackers.results import TrackResults

LIVE_UPDATE_ARGTYPES = [
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

LIVE_UPDATE_WITH_EMBS_ARGTYPES = [
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

AABB_ASSOCIATION_FUNCTIONS = frozenset({"centroid", "ciou", "diou", "giou", "hmiou", "iou"})
OBB_ASSOCIATION_FUNCTIONS = frozenset({"centroid", "diou", "iou"})


def resolve_association_function(cfg: dict[str, Any]) -> str:
    """Normalize and validate a native tracker's association function."""
    asso_func = str(cfg.get("asso_func", "iou") or "iou").strip().lower()
    if asso_func not in AABB_ASSOCIATION_FUNCTIONS:
        available = ", ".join(sorted(AABB_ASSOCIATION_FUNCTIONS))
        raise ValueError(f"Unknown association function {asso_func!r}. Choose from: {available}.")
    cfg["asso_func"] = asso_func
    return asso_func


def association_requires_initial_image(
    cfg: Mapping[str, Any],
    *,
    allow_configured_dimensions: bool = False,
) -> bool:
    """Return whether centroid normalization still needs frame dimensions."""
    if str(cfg.get("asso_func", "iou")).strip().lower() != "centroid":
        return False
    if not allow_configured_dimensions:
        return True
    return int(cfg.get("frame_width", 0) or 0) <= 0 or int(cfg.get("frame_height", 0) or 0) <= 0


def bool_arg(value: Any) -> str:
    return "1" if bool(value) else "0"


def load_tracker_cfg(
    tracker_name: str,
    cfg_dict: dict[str, Any] | None,
    *,
    flatten: bool = False,
) -> dict[str, Any]:
    del flatten  # scalar runtime configs are already flat
    resolved = load_tracker_defaults(tracker_name)
    if cfg_dict is not None:
        resolved.update(cfg_dict)
    return resolved


def ensure_tracker_executable(
    *,
    tracker_name: str,
    display_name: str,
    build_lock: threading.Lock,
    force_rebuild: bool = False,
) -> Path:
    target = f"{tracker_name}_replay"
    return native_common.build_native_target(
        tracker_name=tracker_name,
        display_name=display_name,
        target=target,
        candidates=native_common.candidate_executables(tracker_name),
        force_rebuild=force_rebuild,
        not_found_message=f"Native {display_name} build succeeded but the {target} executable was not found.",
        build_lock=build_lock,
    )


def ensure_tracker_library(
    *,
    tracker_name: str,
    display_name: str,
    build_lock: threading.Lock,
    force_rebuild: bool = False,
) -> Path:
    target = f"{tracker_name}_capi"
    return native_common.build_native_target(
        tracker_name=tracker_name,
        display_name=display_name,
        target=target,
        candidates=native_common.candidate_libraries(tracker_name),
        force_rebuild=force_rebuild,
        not_found_message=f"Native {display_name} build succeeded but the {target} shared library was not found.",
        build_lock=build_lock,
    )


def normalize_detections(dets: np.ndarray | None, *, display_name: str) -> np.ndarray:
    if dets is None:
        det_arr = AABB_SCHEMA.empty_detections()
    else:
        det_arr = np.asarray(dets, dtype=np.float32)
    if det_arr.ndim == 1:
        if det_arr.size == 0:
            det_arr = AABB_SCHEMA.empty_detections()
        else:
            det_arr = det_arr.reshape(1, -1)
    elif det_arr.ndim == 2 and det_arr.shape[1] == 0:
        det_arr = AABB_SCHEMA.empty_detections()
    if det_arr.ndim != 2:
        raise ValueError("Detections must be a 2D array.")
    try:
        schema_from_detection_columns(det_arr.shape[1])
    except ValueError as exc:
        raise NotImplementedError(
            f"Native {display_name} live tracking supports AABB detections with "
            f"{AABB_SCHEMA.detection_cols} columns or OBB detections with {OBB_SCHEMA.detection_cols} columns."
        ) from exc
    return np.ascontiguousarray(det_arr, dtype=np.float32)


def normalize_embeddings(embs: np.ndarray | None, *, rows: int) -> np.ndarray | None:
    if embs is None:
        return None
    emb_arr = np.asarray(embs, dtype=np.float32)
    if emb_arr.size == 0:
        emb_arr = np.empty((0, 0), dtype=np.float32)
    elif emb_arr.ndim == 1:
        emb_arr = emb_arr.reshape(1, -1)
    if emb_arr.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")
    if emb_arr.shape[0] != rows:
        raise ValueError("Detections and embeddings must have the same number of rows.")
    if not np.isfinite(emb_arr).all():
        raise ValueError("Embeddings must contain only finite values.")
    return np.ascontiguousarray(emb_arr, dtype=np.float32)


def normalize_image(img: np.ndarray | None) -> tuple[np.ndarray | None, int]:
    if img is None:
        return None, 0

    img_arr = np.asarray(img)
    if img_arr.dtype != np.uint8:
        img_arr = img_arr.astype(np.uint8, copy=False)
    if img_arr.ndim not in {2, 3}:
        raise ValueError("Image must be a 2D or 3D uint8 array.")
    img_arr = np.ascontiguousarray(img_arr)
    channels = 1 if img_arr.ndim == 2 else int(img_arr.shape[2])
    return img_arr, channels


def call_update(
    update_fn,
    *,
    handle,
    dets: np.ndarray | None,
    img: np.ndarray | None,
    display_name: str,
    last_error,
    embs: np.ndarray | None = None,
    accepts_embeddings: bool = False,
    requires_image: bool = True,
) -> np.ndarray:
    det_arr = normalize_detections(dets, display_name=display_name)
    emb_arr = normalize_embeddings(embs, rows=int(det_arr.shape[0])) if accepts_embeddings else None
    if img is None and requires_image:
        raise ValueError(f"Native {display_name} requires img for live tracking.")
    img_arr, img_channels = normalize_image(img)
    img_ptr = None if img_arr is None or img_arr.size == 0 else ctypes.c_void_p(img_arr.ctypes.data)
    img_rows = 0 if img_arr is None else int(img_arr.shape[0])
    img_cols = 0 if img_arr is None else int(img_arr.shape[1])

    out_capacity = max(int(det_arr.shape[0]), 1)
    out_arr = np.empty((out_capacity, 9), dtype=np.float32)
    out_rows = ctypes.c_int(0)
    out_is_obb = ctypes.c_int(0)
    det_ptr = None if det_arr.size == 0 else ctypes.c_void_p(det_arr.ctypes.data)

    if accepts_embeddings:
        emb_ptr = None if emb_arr is None or emb_arr.size == 0 else ctypes.c_void_p(emb_arr.ctypes.data)
        ok = update_fn(
            handle,
            det_ptr,
            int(det_arr.shape[0]),
            int(det_arr.shape[1]),
            emb_ptr,
            0 if emb_arr is None else int(emb_arr.shape[0]),
            0 if emb_arr is None else int(emb_arr.shape[1]),
            img_ptr,
            img_rows,
            img_cols,
            img_channels,
            ctypes.c_void_p(out_arr.ctypes.data),
            int(out_arr.shape[0]),
            int(out_arr.shape[1]),
            ctypes.byref(out_rows),
            ctypes.byref(out_is_obb),
        )
    else:
        ok = update_fn(
            handle,
            det_ptr,
            int(det_arr.shape[0]),
            int(det_arr.shape[1]),
            img_ptr,
            img_rows,
            img_cols,
            img_channels,
            ctypes.c_void_p(out_arr.ctypes.data),
            int(out_arr.shape[0]),
            int(out_arr.shape[1]),
            ctypes.byref(out_rows),
            ctypes.byref(out_is_obb),
        )

    if ok == 0:
        raise RuntimeError(last_error())

    rows = max(int(out_rows.value), 0)
    input_schema = schema_from_detection_columns(det_arr.shape[1])
    output_schema = get_box_schema_for_mode(bool(out_is_obb.value))
    if input_schema != output_schema:
        raise RuntimeError(
            f"Native {display_name} returned {output_schema.box_type.value} tracks for "
            f"{input_schema.box_type.value} detections."
        )
    return out_arr[:rows, : output_schema.track_cols].copy()


def get_double_result(library, symbol: str, handle, last_error) -> float:
    fn = getattr(library, symbol, None)
    if fn is None:
        return 0.0
    out_value = ctypes.c_double(0.0)
    ok = fn(handle, ctypes.byref(out_value))
    if ok == 0:
        raise RuntimeError(last_error())
    return float(out_value.value)


class NativeTrackerMixin:
    _native_display_name: str
    _tracks_reid_timing = False

    def requires_image(
        self,
        dets: np.ndarray,
        embs: np.ndarray | None = None,
        masks: np.ndarray | None = None,
    ) -> bool:
        """Return whether this native tracker consumes images."""
        del dets, embs, masks
        return bool(self.uses_img)

    def _init_native_handle(self, *, library, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self._library = library
        self._handle = self._library.create(self.cfg)
        self._det_cols: int | None = None
        self.class_catalog = ClassCatalog()
        self.class_ids = self.class_catalog.class_ids
        self.class_names = self.class_catalog.names
        self._reset_reid_timing()

    def configure_class_catalog(
        self,
        *,
        class_ids: Iterable[int] | None = None,
        class_names: Mapping[int, str] | None = None,
    ) -> None:
        """Configure and enforce the detector class contract in Python."""
        self.class_catalog = ClassCatalog.from_metadata(class_ids=class_ids, class_names=class_names)
        self.class_ids = self.class_catalog.class_ids
        self.class_names = self.class_catalog.names

    def _reset_reid_timing(self) -> None:
        self.last_reid_time_ms = 0.0
        self.last_reid_preprocess_time_ms = 0.0
        self.last_reid_process_time_ms = 0.0
        self.last_reid_postprocess_time_ms = 0.0

    def _coerce_detections_for_mode(self, dets: np.ndarray | None) -> np.ndarray:
        det_arr = np.asarray(dets) if dets is not None else np.empty((0, 0), dtype=np.float32)
        if det_arr.ndim == 1 and det_arr.size:
            det_arr = det_arr.reshape(1, -1)

        has_explicit_layout = det_arr.ndim == 2 and det_arr.shape[1] in {
            AABB_SCHEMA.detection_cols,
            OBB_SCHEMA.detection_cols,
        }
        if det_arr.size or has_explicit_layout:
            if not has_explicit_layout:
                raise ValueError(
                    f"Native {self._native_display_name} expects AABB detections with "
                    f"{AABB_SCHEMA.detection_cols} columns or OBB detections with "
                    f"{OBB_SCHEMA.detection_cols} columns."
                )
            if det_arr.size and not np.isfinite(det_arr).all():
                raise ValueError("Native tracker detections must contain only finite values.")

            incoming_cols = int(det_arr.shape[1])
            if self._det_cols is not None and incoming_cols != self._det_cols:
                raise ValueError(
                    f"Native {self._native_display_name} tracker cannot switch between "
                    "AABB and OBB inputs after initialization."
                )

            schema = schema_from_detection_columns(det_arr.shape[1])
            asso_func = str(getattr(self, "cfg", {}).get("asso_func", "iou"))
            if schema.is_obb and asso_func not in OBB_ASSOCIATION_FUNCTIONS:
                available = ", ".join(sorted(OBB_ASSOCIATION_FUNCTIONS))
                raise ValueError(
                    f"Association function {asso_func!r} has no oriented-box implementation. "
                    f"Choose from: {available}."
                )
            layout = get_detection_layout(schema.is_obb)
            if det_arr.size:
                boxes = layout.boxes(det_arr)
                if layout.is_obb:
                    if np.any(boxes[:, 2:4] <= 0):
                        raise ValueError("Native OBB detections must have positive width and height.")
                elif np.any(boxes[:, 2] <= boxes[:, 0]) or np.any(boxes[:, 3] <= boxes[:, 1]):
                    raise ValueError("Native AABB detections must satisfy x2 > x1 and y2 > y1.")
                classes = layout.classes(det_arr)
                if not np.equal(classes, np.floor(classes)).all():
                    raise ValueError("Detector class IDs must be integers.")
                self.class_catalog.validate_detections(det_arr, layout)

            if self._det_cols is None:
                self._det_cols = incoming_cols
        elif det_arr.ndim == 2 and det_arr.shape[1] != 0:
            raise ValueError(
                f"Native {self._native_display_name} expects empty AABB detections with "
                f"{AABB_SCHEMA.detection_cols} columns or empty OBB detections with "
                f"{OBB_SCHEMA.detection_cols} columns."
            )
        elif self._det_cols is not None:
            det_arr = np.empty((0, self._det_cols), dtype=np.float32)
        return np.ascontiguousarray(det_arr, dtype=np.float32)

    def _normalize_tracks_for_mode(self, tracks: np.ndarray) -> TrackResults:
        """Validate native output against the mode latched from detections."""
        detection_cols = self._det_cols if self._det_cols is not None else AABB_SCHEMA.detection_cols
        schema = schema_from_detection_columns(detection_cols)
        return TrackResults(tracks, schema=schema)

    def _refresh_reid_timings(self) -> None:
        if not self._tracks_reid_timing:
            return
        getter = getattr(self._library, "get_last_reid_time_ms", None)
        self.last_reid_time_ms = float(getter(self._handle)) if callable(getter) else 0.0
        for phase in ("preprocess", "process", "postprocess"):
            phase_getter = getattr(self._library, f"get_last_reid_{phase}_time_ms", None)
            value = float(phase_getter(self._handle)) if callable(phase_getter) else 0.0
            setattr(self, f"last_reid_{phase}_time_ms", value)

    def reset(self) -> None:
        self._library.reset(self._handle)
        self._det_cols = None
        self._reset_reid_timing()

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


def build_replay_command(
    *,
    executable: Path,
    mot_root: str,
    det_emb_root: Path,
    detector_name: str,
    seq_name: str,
    exp_folder: str,
    conf_threshold: float,
    target_fps: int | None,
    extra_args: Iterable[str] = (),
) -> list[str]:
    detector_key = Path(detector_name).stem if Path(detector_name).suffix else str(detector_name)
    output_path = Path(exp_folder) / f"{seq_name}.txt"
    return [
        str(executable),
        "--mot-root",
        str(mot_root),
        "--det-emb-root",
        str(det_emb_root),
        "--detector-name",
        detector_key,
        *extra_args,
        "--sequence",
        seq_name,
        "--output",
        str(output_path),
        "--conf-threshold",
        str(float(conf_threshold)),
        "--target-fps",
        str(int(target_fps or 0)),
    ]


def run_replay_process(
    *,
    cmd: list[str],
    seq_name: str,
    display_name: str,
    progress_queue,
    subprocess_module,
    env: dict[str, str] | None = None,
) -> tuple[str, list[int], dict[str, Any]]:
    popen_kwargs: dict[str, Any] = {
        "stdout": subprocess_module.PIPE,
        "stderr": subprocess_module.PIPE,
        "text": True,
        "bufsize": 1,
    }
    if env is not None:
        popen_kwargs["env"] = env
    process = subprocess_module.Popen(cmd, **popen_kwargs)
    stderr_lines: list[str] = []
    stderr_thread = threading.Thread(
        target=native_common.drain_native_stderr,
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
            f"Native {display_name} replay failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"{stderr_text or stdout_text.strip() or f'Unknown native {display_name} failure.'}"
        )

    summary = native_common.parse_summary(stdout_text, display_name=display_name)
    if str(summary.get("sequence")) != str(seq_name):
        raise RuntimeError(
            f"Native {display_name} summary sequence mismatch: expected {seq_name!r}, got {summary.get('sequence')!r}."
        )

    kept_ids = [int(frame_id) for frame_id in summary.get("kept_frame_ids", [])]
    timing = {
        "track_time_ms": float(summary.get("track_time_ms", 0.0)),
        "num_frames": int(summary.get("num_frames", len(kept_ids))),
    }
    return seq_name, kept_ids, timing


def limited_replay_env() -> dict[str, str]:
    env = dict(os.environ)
    if not env.get("BOXMOT_NATIVE_CV_THREADS"):
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        env.setdefault("NUMEXPR_NUM_THREADS", "1")
        env.setdefault("OPENCV_FOR_THREADS_NUM", "1")
    return env
