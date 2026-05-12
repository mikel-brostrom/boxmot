from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

import cv2

from boxmot.configs import get_mode_default
from boxmot.engine.results import Results
from boxmot.engine.workflow_results import TrackRunResult
from boxmot.trackers.track_results import TrackResults
from boxmot.trackers.tracker_zoo import TRACKER_MAPPING, create_tracker, get_tracker_config
from boxmot.utils.mot_utils import convert_to_mmot_obb_format, convert_to_mot_format, write_mot_results
from boxmot.utils.rich.reporting import primary_model_ref as _primary_model_ref
from boxmot.utils.rich.pipeline import PipelineTracker
from boxmot.utils.rich.track_reporting import (
    TRACK_RUN_STEP,
    TRACK_SETUP_STEP,
    TrackWorkflowReporter,
)
from boxmot.utils.timing import TimingStats, wrap_tracker_reid
from boxmot.utils.torch_utils import select_device
import boxmot.utils.rich.ui as ui
from boxmot.engine.workflow_support import (
    build_detector_from_spec,
    build_tracker_from_spec,
    build_tracker_with_reid_spec,
    reid_path_from_spec,
    resolve_output_fps,
    resolve_track_output_dir,
    save_video,
    tracker_name_from_spec,
)
from boxmot.utils.misc import suppress_boxmot_logs


def _is_live_source(source: Any) -> bool:
    if isinstance(source, int):
        return True
    if isinstance(source, str):
        return source.isdigit() or "://" in source
    return False


def Boxmot(*args, **kwargs):
    """Lazy compatibility loader for the public Boxmot facade."""
    from boxmot import Boxmot as PublicBoxmot

    return PublicBoxmot(*args, **kwargs)


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
        reid_preprocess: str | None = None,
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
            reid_preprocess=reid_preprocess,
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
        tr = TrackResults(TrackerRuntime._ensure_2d_tracks(tracks))
        if tr.size == 0:
            return np.empty((0, 0), dtype=np.float32)
        if tr.is_obb:
            return convert_to_mmot_obb_format(tr, frame_idx)
        return convert_to_mot_format(tr, frame_idx)

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
    for _ in result.results:
        pass
    result.refresh()


class TrackingSession:
    """Compatibility wrapper around the public Python API tracking facade."""

    def __init__(self, args):
        self.args = args

    def _should_consume_result(self) -> bool:
        return _should_consume_result(self.args)

    def _resolve_output_stem(self) -> str:
        from boxmot.engine.workflow_support import resolve_output_stem

        return resolve_output_stem(getattr(self.args, "source", ""))

    def _resolve_output_fps(self) -> int:
        fps = getattr(self.args, "fps", None)
        if fps is not None:
            return int(fps)
        return int(resolve_output_fps(getattr(self.args, "source", None)))

    @staticmethod
    def initialize_trackers(predictor, args):
        tracker_spec = getattr(args, "tracker", "")
        tracker_name = tracker_name_from_spec(tracker_spec, required=True)
        if tracker_name not in TRACKER_MAPPING:
            available = ", ".join(sorted(TRACKER_MAPPING))
            raise ValueError(f"'{tracker_name}' is not supported. Supported ones are {available}")

        reid_weights = _primary_model_ref(getattr(args, "reid", None))
        if reid_weights is not None:
            reid_weights = Path(reid_weights)

        batch_size = int(getattr(getattr(predictor, "dataset", None), "bs", 1) or 1)
        predictor.trackers = [
            build_tracker_from_spec(
                tracker_spec,
                device=getattr(predictor, "device", "cpu"),
                half=bool(getattr(args, "half", False)),
                tracker_backend=getattr(args, "tracker_backend", None),
                reid_weights=reid_weights,
                reid_preprocess=getattr(args, "reid_preprocess", None),
            )
            for _ in range(batch_size)
        ]
        target_id = getattr(args, "target_id", None)
        if target_id is not None:
            for tracker in predictor.trackers:
                setattr(tracker, "target_id", target_id)
        return predictor.trackers

    def run(self):
        boxmot = Boxmot(
            detector=_primary_model_ref(getattr(self.args, "detector", None)),
            reid=_primary_model_ref(getattr(self.args, "reid", None)),
            tracker=getattr(self.args, "tracker", get_mode_default("track", "tracker")),
            classes=getattr(self.args, "classes", None),
            project=getattr(self.args, "project", get_mode_default("track", "project")),
        )
        result = boxmot.track(
            source=getattr(self.args, "source", get_mode_default("track", "source")),
            imgsz=getattr(self.args, "imgsz", None),
            conf=getattr(self.args, "conf", None),
            iou=float(getattr(self.args, "iou", get_mode_default("track", "iou"))),
            device=getattr(self.args, "device", get_mode_default("track", "device")),
            half=bool(getattr(self.args, "half", get_mode_default("track", "half"))),
            save=bool(getattr(self.args, "save", False)),
            save_txt=bool(getattr(self.args, "save_txt", False)),
            show=bool(getattr(self.args, "show", False)),
            verbose=bool(getattr(self.args, "verbose", False)),
            tracker_backend=getattr(self.args, "tracker_backend", None),
        )
        if getattr(self.args, "show", False):
            result.show()
        elif self._should_consume_result():
            _consume_run(result)
        return result


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
        tracker_backend=getattr(args, "tracker_backend", None),
        reid_weights=reid_weights,
        reid_preprocess=getattr(args, "reid_preprocess", None),
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
    show_trajectories: bool = False,
    pipeline: PipelineTracker | None = None,
) -> TrackRunResult:
    source = getattr(args, "source", get_mode_default("track", "source"))
    verbose = bool(getattr(args, "verbose", get_mode_default("track", "verbose")))

    with suppress_boxmot_logs((not verbose) or pipeline is not None, level="WARNING"):
        detector_runtime = detector if detector is not None else _build_detector(args, detector_spec, classes)
        tracker_runtime = tracker if tracker is not None else _build_tracker(args, tracker_spec)
        reid_runtime = reid if reid is not None else _build_reid(args, tracker_runtime, reid_spec, tracker_spec)

    if show_trajectories and drawer is None:
        drawer = lambda frame, tracks: tracker_runtime.plot_results(frame, show_trajectories=True)

    if pipeline is not None:
        pipeline.advance("Starting tracker...")

    run = Results(
        source,
        detector_runtime,
        reid_runtime,
        tracker_runtime,
        verbose=verbose and pipeline is None,
        drawer=drawer,
        progress_callback=pipeline.callback() if pipeline is not None else None,
    )

    output_dir = resolve_track_output_dir(Path(getattr(args, "project", "runs")), source)
    text_path = output_dir / "tracks.txt" if bool(getattr(args, "save_txt", False)) else None
    video_path = output_dir / "tracks.mp4" if bool(getattr(args, "save", False)) else None

    show = bool(getattr(args, "show", False))
    needs_iteration = text_path is not None or video_path is not None or show

    if needs_iteration:
        if text_path is not None:
            text_path.parent.mkdir(parents=True, exist_ok=True)
            if text_path.exists():
                text_path.unlink()
        video_writer = None
        video_fps = resolve_output_fps(source) if video_path is not None else 30.0
        if video_path is not None:
            video_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            for frame_result in run:
                if text_path is not None:
                    write_mot_results(text_path, frame_result.to_mot())
                if video_path is not None:
                    rendered = frame_result.render()
                    if video_writer is None:
                        h, w = rendered.shape[:2]
                        video_writer = cv2.VideoWriter(
                            str(video_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            video_fps,
                            (w, h),
                        )
                    video_writer.write(rendered)
                if show:
                    if not frame_result.show():
                        break
        finally:
            if video_writer is not None:
                video_writer.release()
            if show:
                cv2.destroyAllWindows()

    result = TrackRunResult(
        source=source,
        results=run,
        video_path=video_path,
        text_path=text_path,
    )
    if not needs_iteration and _should_consume_result(args):
        _consume_run(result)
    if pipeline is not None:
        result.refresh()
        pipeline.complete_step()
        if int(result.summary.get("frames", 0)) > 0:
            pipeline.set_detail_renderable("Summary", result.renderable())
        else:
            pipeline.update("No frames processed.")
    return result


def main(args):
    pipeline = TrackWorkflowReporter(args).pipeline()
    with pipeline:
        return run_track(
            args,
            detector_spec=_primary_model_ref(getattr(args, "detector", None)),
            reid_spec=_primary_model_ref(getattr(args, "reid", None)),
            tracker_spec=getattr(args, "tracker", None),
            classes=getattr(args, "classes", None),
            pipeline=pipeline,
        )
