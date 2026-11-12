"""Detector-free EdgeTAM tracking with interactive box and point prompts."""

from __future__ import annotations

import time
from contextlib import ExitStack
from copy import copy
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from boxmot.engine.tracking.runner import FrameTiming, RunSummary
from boxmot.engine.tracking.sinks import TrackSink, _StopTrackingRequested, render_result
from boxmot.engine.tracking.sources import FrameSource, create_frame_source
from boxmot.engine.ui.prompts import PromptController, PromptWindow
from boxmot.pipelines import PipelineResult
from boxmot.structures import Boxes, Detections, Frame

if TYPE_CHECKING:
    from boxmot.engine.tracking.attributes import AttributeRun
    from boxmot.engine.tracking.workflow import TrackRun
    from boxmot.engine.ui.workflow.pipeline import PipelineTracker


def _create_prompted_tracker(args: Any) -> Any:
    """Resolve the tracker only after a desktop window has opened successfully."""
    from boxmot import EdgeTAMModelConfig, create_tracker
    from boxmot.engine.tracking.workflow import _tracker_spec

    return create_tracker(
        _tracker_spec(args, "aabb"),
        prompt_model=EdgeTAMModelConfig(
            checkpoint=getattr(args, "prompt_weights", None) or "edgetam.pt",
            device=str(getattr(args, "device", None) or "cpu"),
        ),
    )


def _empty_detections(frame: Frame) -> Detections:
    """Keep the canonical tracker boundary while running no detector."""
    return Detections(
        Boxes(torch.empty((0, 4), dtype=torch.float32)),
        torch.empty(0, dtype=torch.float32),
        torch.empty(0, dtype=torch.int64),
        sample_id=frame.sample_id,
    )


def _apply_prompts(controller: PromptController, tracker: Any) -> None:
    """Queue complete corrections before advancing their displayed frame."""
    for object_id in sorted(controller.removed):
        tracker.remove_object(object_id)
    for object_id, prompt in sorted(controller.pending.items()):
        if prompt.box is not None:
            tracker.add_box(np.asarray(prompt.box, dtype=np.float32), object_id=object_id)
        if prompt.points:
            tracker.add_points(
                np.asarray(prompt.points, dtype=np.float32),
                labels=np.asarray(prompt.labels, dtype=np.int64),
                object_id=object_id,
            )


def run_prompted_track(
    args: Any,
    *,
    tracker: Any | None = None,
    source: FrameSource | None = None,
    sinks: tuple[TrackSink, ...] | None = None,
    ui_pipeline: PipelineTracker | None = None,
    attribute_run: AttributeRun | None = None,
) -> TrackRun:
    """Prompt each sequence, propagate masks, and share rendered output with sinks."""
    from boxmot.engine.tracking.workflow import TrackRun, _default_sinks

    if str(getattr(args, "geometry", "auto") or "auto") not in {"auto", "aabb"}:
        raise ValueError("Interactive EdgeTAM tracking requires AABB geometry.")
    args.geometry = "aabb"
    if str(getattr(args, "tracker_backend", "python") or "python") != "python":
        raise ValueError("Interactive EdgeTAM tracking requires the Python backend.")
    if bool(getattr(args, "per_class", False)):
        raise ValueError("Interactive EdgeTAM tracking does not use per-class tracking.")

    started = time.perf_counter()
    timings: list[FrameTiming] = []
    frame_count = sequence_count = track_rows = 0
    identities: set[tuple[str, int]] = set()
    interrupted = False
    video_path = mot_path = json_path = None
    window = PromptWindow()
    active_sequence: str | None = None
    active_size: tuple[int, int] | None = None
    try:
        with ExitStack() as resources:
            # Register supplied resources before GUI checks, so failure also
            # closes sources/sinks whose ownership the caller handed us.
            if source is not None:
                resources.callback(source.close)
            if sinks is not None:
                for sink in sinks:
                    resources.callback(sink.close)
            resources.callback(window.close)
            if ui_pipeline is not None:
                ui_pipeline.update("Opening video prompt window…")
            window.open()
            if tracker is None:
                tracker = _create_prompted_tracker(args)
            resources.callback(tracker.reset)
            window.controller.max_objects = int(getattr(getattr(tracker, "config", None), "max_objects", 32))
            if source is None:
                source = create_frame_source(
                    getattr(args, "source", "0"),
                    stride=int(getattr(args, "vid_stride", 1) or 1),
                    camera_warmup_s=0.0,
                    latest_camera_frame=True,
                )
                resources.callback(source.close)
            if sinks is None:
                output_args = copy(args)
                output_args.show = False  # The prompt window owns the preview.
                sinks, video_path, mot_path, json_path = _default_sinks(output_args)
                for sink in sinks:
                    resources.callback(sink.close)
            iterator = iter(source)
            if callable(getattr(iterator, "close", None)):
                resources.callback(iterator.close)
            if ui_pipeline is not None:
                ui_pipeline.advance("Select objects in the video window…")
            for frame in iterator:
                sequence = frame.sequence_id or "prompted-source"
                if sequence != active_sequence or frame.image_size != active_size:
                    tracker.reset()
                    if attribute_run is not None:
                        attribute_run.enricher.reset()
                    window.controller.reset(live=bool(frame.source_uri and frame.source_uri.startswith("camera:")))
                    active_sequence, active_size = sequence, frame.image_size
                    sequence_count += 1
                if ui_pipeline is not None and (window.controller.paused or window.controller.choosing_frame):
                    action = (
                        "Choose a camera frame with Space / Enter"
                        if window.controller.choosing_frame
                        else "Select objects"
                    )
                    ui_pipeline.update(f"{action} in the video window • {frame.sample_id}")
                if not window.prepare_frame(frame):
                    interrupted = True
                    break
                if window.controller.choosing_frame:
                    continue
                if ui_pipeline is not None and window.controller.pending:
                    ui_pipeline.update("Processing selected objects…")
                frame_started = time.perf_counter()
                _apply_prompts(window.controller, tracker)
                detections = _empty_detections(frame)
                tracks = tracker.update(detections, frame=frame)
                window.controller.commit()
                attributes = {} if attribute_run is None else attribute_run.enricher.update(frame, tracks)
                result = PipelineResult(detections, tracks, attributes)
                if attribute_run is not None:
                    attribute_run.poll()
                image = render_result(
                    frame, result, line_width=int(getattr(args, "line_width", 2) or 2),
                    show_labels=bool(getattr(args, "show_labels", True)),
                )
                for sink in sinks:
                    write_rendered = getattr(sink, "write_rendered", None)
                    if callable(write_rendered):
                        write_rendered(frame, result, image)
                    else:
                        sink.write(frame, result)
                frame_count += 1
                track_rows += len(tracks)
                identities.update(
                    (f"{sequence}:{sequence_count}", int(object_id)) for object_id in tracks.track_ids.tolist()
                )
                timings.append(FrameTiming(frame.sample_id, (time.perf_counter() - frame_started) * 1000))
                if ui_pipeline is not None:
                    detail = f"Processed {frame_count} frame(s) • {frame.sample_id}"
                    if attribute_run is not None:
                        detail += f"\n{attribute_run.status_summary}"
                    ui_pipeline.update(detail)
                window.show_result(image)
                if window.controller.quit:
                    interrupted = True
                    break
    except (KeyboardInterrupt, _StopTrackingRequested):
        interrupted = True
    return TrackRun(
        RunSummary(
            frames=frame_count,
            sequences=sequence_count,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            timings=tuple(timings),
            interrupted=interrupted,
            detections=0,
            track_rows=track_rows,
            unique_track_ids=len(identities),
        ),
        video_path=video_path,
        mot_path=mot_path,
        json_path=json_path,
    )
