from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from boxmot.configs import get_mode_default
from boxmot.engine.tracking.mot import write_mot_results
from boxmot.engine.tracking.results import Results
from boxmot.engine.workflows.results import TrackRunResult
from boxmot.engine.workflows.support import (
    build_detector_from_spec,
    build_tracker_from_spec,
    build_tracker_with_reid_spec,
    reid_path_from_spec,
    resolve_tracker_class_metadata,
    resolve_output_fps,
    resolve_track_output_dir,
    tracker_reid_model_from_spec,
)
from boxmot.utils.misc import suppress_boxmot_logs
from boxmot.utils.rich.reporters.track import TRACK_RUN_STEP, TRACK_SETUP_STEP, TrackWorkflowReporter
from boxmot.utils.rich.workflow.fields import first_value
from boxmot.utils.rich.workflow.pipeline import PipelineTracker


def _is_live_source(source: Any) -> bool:
    if isinstance(source, int):
        return True
    if isinstance(source, str):
        return source.isdigit() or "://" in source
    return False


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


def _build_detector(
    args,
    detector_spec: Any,
    classes: list[int] | None,
    detector_kwargs: dict[str, Any] | None,
):
    spec = detector_spec if detector_spec is not None else first_value(getattr(args, "detector", None))
    return build_detector_from_spec(
        spec,
        classes=classes,
        device=getattr(args, "device", get_mode_default("track", "device")),
        imgsz=getattr(args, "imgsz", None),
        conf=getattr(args, "conf", None),
        iou=float(getattr(args, "iou", get_mode_default("track", "iou"))),
        detector_kwargs=detector_kwargs,
    )


def _build_tracker(
    args,
    tracker_spec: Any,
    reid_spec: Any,
    *,
    class_ids: tuple[int, ...] | None = None,
    class_names: dict[int, str] | None = None,
    reid_kwargs: dict[str, Any] | None = None,
    tracker_kwargs: dict[str, Any] | None = None,
):
    spec = tracker_spec if tracker_spec is not None else getattr(args, "tracker", get_mode_default("track", "tracker"))
    resolved_reid_spec = reid_spec if reid_spec is not None else first_value(getattr(args, "reid", None))
    reid_weights = reid_path_from_spec(resolved_reid_spec, required=False)
    reid_model = tracker_reid_model_from_spec(resolved_reid_spec)
    runtime_reid_kwargs = dict(reid_kwargs or {})
    if "preprocess_name" in runtime_reid_kwargs:
        raise ValueError("Unsupported reid_kwargs: preprocess_name. Use preprocess=... instead.")
    unknown_reid_kwargs = set(runtime_reid_kwargs) - {"device", "half", "preprocess"}
    if unknown_reid_kwargs:
        unknown = ", ".join(sorted(unknown_reid_kwargs))
        raise ValueError(f"Unknown reid_kwargs: {unknown}")
    reid_device = runtime_reid_kwargs.get("device", getattr(args, "device", get_mode_default("track", "device")))
    reid_half = runtime_reid_kwargs.get("half", bool(getattr(args, "half", get_mode_default("track", "half"))))
    reid_preprocess = runtime_reid_kwargs.get("preprocess", getattr(args, "reid_preprocess", None))
    return build_tracker_from_spec(
        spec,
        device=reid_device,
        half=bool(reid_half),
        tracker_backend=getattr(args, "tracker_backend", None),
        reid_weights=reid_weights,
        reid_model=reid_model,
        reid_preprocess=reid_preprocess,
        class_ids=class_ids,
        class_names=class_names,
        tracker_kwargs=tracker_kwargs,
    )


def _build_reid(
    args,
    tracker: Any,
    reid_spec: Any,
    tracker_spec: Any,
    reid_kwargs: dict[str, Any] | None,
):
    return build_tracker_with_reid_spec(
        tracker_spec if tracker_spec is not None else getattr(args, "tracker", get_mode_default("track", "tracker")),
        tracker,
        reid_spec if reid_spec is not None else first_value(getattr(args, "reid", None)),
        device=getattr(args, "device", get_mode_default("track", "device")),
        half=bool(getattr(args, "half", get_mode_default("track", "half"))),
        reid_kwargs=reid_kwargs,
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
    detector_kwargs: dict[str, Any] | None = None,
    reid_kwargs: dict[str, Any] | None = None,
    tracker_kwargs: dict[str, Any] | None = None,
    drawer=None,
    show_trajectories: bool = False,
    pipeline: PipelineTracker | None = None,
) -> TrackRunResult:
    source = getattr(args, "source", get_mode_default("track", "source"))
    verbose = bool(getattr(args, "verbose", get_mode_default("track", "verbose")))

    with suppress_boxmot_logs((not verbose) or pipeline is not None, level="WARNING"):
        detector_runtime = (
            detector
            if detector is not None
            else _build_detector(args, detector_spec, classes, detector_kwargs)
        )
        class_ids, class_names = resolve_tracker_class_metadata(args, detector_runtime)
        tracker_runtime = (
            tracker
            if tracker is not None
            else _build_tracker(
                args,
                tracker_spec,
                reid_spec,
                class_ids=class_ids,
                class_names=class_names,
                reid_kwargs=reid_kwargs,
                tracker_kwargs=tracker_kwargs,
            )
        )
        if tracker is not None and hasattr(tracker_runtime, "configure_class_catalog"):
            tracker_runtime.configure_class_catalog(class_ids=class_ids, class_names=class_names)
        reid_runtime = (
            reid
            if reid is not None
            else _build_reid(args, tracker_runtime, reid_spec, tracker_spec, reid_kwargs)
        )

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
            # Flush Online GTA interpolated entries to MOT results file
            if text_path is not None and hasattr(run, "tracker"):
                _trk = getattr(run, "tracker", None)
                if _trk is not None and hasattr(_trk, "flush_gta"):
                    gta_entries = _trk.flush_gta()
                    if gta_entries.size:
                        write_mot_results(text_path, gta_entries)
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
            detector_spec=first_value(getattr(args, "detector", None)),
            reid_spec=first_value(getattr(args, "reid", None)),
            tracker_spec=getattr(args, "tracker", None),
            classes=getattr(args, "classes", None),
            pipeline=pipeline,
        )


__all__ = (
    "TRACK_RUN_STEP",
    "TRACK_SETUP_STEP",
    "_consume_run",
    "_should_consume_result",
    "main",
    "run_track",
)
