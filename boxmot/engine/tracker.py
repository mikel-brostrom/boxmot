from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from boxmot.configs import get_mode_default
from boxmot.engine.results import Results
from boxmot.engine.workflow_results import TrackRunResult
from boxmot.trackers.tracker_zoo import TRACKER_MAPPING, create_tracker, get_tracker_config
from boxmot.utils.mot_utils import convert_to_mmot_obb_format, convert_to_mot_format
from boxmot.utils.timing import TimingStats, wrap_tracker_reid
from boxmot.engine.workflow_support import (
    build_detector_from_spec,
    build_tracker_from_spec,
    build_tracker_with_reid_spec,
    reid_path_from_spec,
    resolve_output_fps,
    resolve_track_output_dir,
    save_video,
    suppress_boxmot_logs,
)


def _primary_model_ref(value):
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def _is_live_source(source: Any) -> bool:
    if isinstance(source, int):
        return True
    if isinstance(source, str):
        return source.isdigit() or "://" in source
    return False


class TrackerRuntime:
    """Wrap one tracker instance with timing and formatting helpers."""

    def __init__(self, tracker: Any, timing_stats: TimingStats | None = None) -> None:
        self.tracker = tracker
        self.timing_stats = timing_stats

    @classmethod
    def create(
        cls,
        tracker_name: str,
        reid_weights,
        device,
        half: bool,
        per_class: bool,
        evolve_param_dict: dict | None = None,
        target_id: int | None = None,
        timing_stats: TimingStats | None = None,
    ) -> "TrackerRuntime":
        normalized_tracker = str(tracker_name).lower()
        if normalized_tracker not in TRACKER_MAPPING:
            available = ", ".join(sorted(TRACKER_MAPPING))
            raise ValueError(f"'{tracker_name}' is not supported. Supported ones are {available}")

        tracker = create_tracker(
            tracker_type=normalized_tracker,
            tracker_config=get_tracker_config(normalized_tracker),
            reid_weights=reid_weights,
            device=device,
            half=half,
            per_class=per_class,
            evolve_param_dict=evolve_param_dict,
        )
        if target_id is not None:
            tracker.target_id = target_id
        if timing_stats is not None:
            wrap_tracker_reid(tracker, timing_stats)
        return cls(tracker, timing_stats=timing_stats)

    @staticmethod
    def _ensure_2d_tracks(tracks: np.ndarray) -> np.ndarray:
        arr = np.asarray(tracks, dtype=np.float32)
        if arr.size == 0:
            if arr.ndim == 2:
                return arr
            return np.empty((0, 0), dtype=np.float32)
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        return arr

    @staticmethod
    def format_for_mot(tracks: np.ndarray, frame_idx: int) -> np.ndarray:
        arr = TrackerRuntime._ensure_2d_tracks(tracks)
        if arr.size == 0:
            return np.empty((0, 0), dtype=np.float32)
        if arr.shape[1] >= 9:
            return convert_to_mmot_obb_format(arr, frame_idx)
        return convert_to_mot_format(arr, frame_idx)

    @property
    def names(self):
        return getattr(self.tracker, "names", None)

    @names.setter
    def names(self, value) -> None:
        setattr(self.tracker, "names", value)

    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray | None = None) -> tuple[np.ndarray, float]:
        elapsed_ms = 0.0
        started = False
        if self.timing_stats is not None:
            self.timing_stats.reset_frame_reid()
            self.timing_stats.start_tracking()
            started = True
        else:
            start_time = time.perf_counter()

        try:
            if embs is None:
                tracks = self.tracker.update(dets, img)
            else:
                try:
                    tracks = self.tracker.update(dets, img, embs)
                except TypeError:
                    tracks = self.tracker.update(dets, img)
        finally:
            if started:
                self.timing_stats.end_tracking()
                elapsed_ms = self.timing_stats.get_last_track_time()
            else:
                elapsed_ms = (time.perf_counter() - start_time) * 1000

        return self._ensure_2d_tracks(tracks), elapsed_ms

    def plot_results(
        self,
        img: np.ndarray,
        show_trajectories: bool,
        *,
        thickness: int = 2,
        show_kf_preds: bool = False,
    ) -> np.ndarray:
        if hasattr(self.tracker, "plot_results"):
            return self.tracker.plot_results(
                img,
                show_trajectories,
                thickness=thickness,
                show_kf_preds=show_kf_preds,
            )
        return img

    def __getattr__(self, name: str):
        return getattr(self.tracker, name)


def _should_consume_result(args) -> bool:
    if getattr(args, "show", False):
        return False
    if getattr(args, "save", False) or getattr(args, "save_txt", False):
        return False
    return not _is_live_source(getattr(args, "source", None))


def _consume_run(result: TrackRunResult) -> None:
    previous_cache_results = getattr(result.results, "_cache_results", True)
    try:
        result.results._cache_results = False
        for _ in result.results:
            pass
    finally:
        result.results._cache_results = previous_cache_results
    result.refresh()


def _build_detector(args, detector_spec: Any, classes: list[int] | None):
    spec = detector_spec if detector_spec is not None else _primary_model_ref(getattr(args, "detector", None))
    return build_detector_from_spec(
        spec,
        classes=classes,
        device=getattr(args, "device", get_mode_default("track", "device")),
        imgsz=getattr(args, "imgsz", None),
        conf=getattr(args, "conf", None),
        iou=float(getattr(args, "iou", get_mode_default("track", "iou"))),
    )


def _build_tracker(args, tracker_spec: Any):
    spec = tracker_spec if tracker_spec is not None else getattr(args, "tracker", get_mode_default("track", "tracker"))
    reid_weights = reid_path_from_spec(_primary_model_ref(getattr(args, "reid", None)), required=False)
    return build_tracker_from_spec(
        spec,
        device=getattr(args, "device", get_mode_default("track", "device")),
        half=bool(getattr(args, "half", get_mode_default("track", "half"))),
        reid_weights=reid_weights,
    )


def _build_reid(args, tracker: Any, reid_spec: Any, tracker_spec: Any):
    return build_tracker_with_reid_spec(
        tracker_spec if tracker_spec is not None else getattr(args, "tracker", get_mode_default("track", "tracker")),
        tracker,
        reid_spec if reid_spec is not None else _primary_model_ref(getattr(args, "reid", None)),
        device=getattr(args, "device", get_mode_default("track", "device")),
        half=bool(getattr(args, "half", get_mode_default("track", "half"))),
    )


def run_track(
    args,
    *,
    detector=None,
    reid=None,
    tracker=None,
    detector_spec: Any = None,
    reid_spec: Any = None,
    tracker_spec: Any = None,
    classes: list[int] | None = None,
    drawer=None,
) -> TrackRunResult:
    source = getattr(args, "source", get_mode_default("track", "source"))
    verbose = bool(getattr(args, "verbose", get_mode_default("track", "verbose")))

    with suppress_boxmot_logs(not verbose, level="WARNING"):
        detector_runtime = detector if detector is not None else _build_detector(args, detector_spec, classes)
        tracker_runtime = tracker if tracker is not None else _build_tracker(args, tracker_spec)
        reid_runtime = reid if reid is not None else _build_reid(args, tracker_runtime, reid_spec, tracker_spec)

    run = Results(source, detector_runtime, reid_runtime, tracker_runtime, verbose=verbose, drawer=drawer)

    output_dir = resolve_track_output_dir(Path(getattr(args, "project", "runs")), source)
    text_path = output_dir / "tracks.txt" if bool(getattr(args, "save_txt", False)) else None
    video_path = output_dir / "tracks.mp4" if bool(getattr(args, "save", False)) else None

    if text_path is not None:
        run.save(text_path)
    if video_path is not None:
        save_video(run, video_path, fps=resolve_output_fps(source))

    result = TrackRunResult(
        source=source,
        results=run,
        video_path=video_path,
        text_path=text_path,
    )
    if bool(getattr(args, "show", False)):
        result.show()
    elif _should_consume_result(args):
        _consume_run(result)
    return result


def main(args):
    return run_track(
        args,
        detector_spec=_primary_model_ref(getattr(args, "detector", None)),
        reid_spec=_primary_model_ref(getattr(args, "reid", None)),
        tracker_spec=getattr(args, "tracker", None),
        classes=getattr(args, "classes", None),
    )
