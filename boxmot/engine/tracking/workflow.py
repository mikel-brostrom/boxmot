"""Canonical live-tracking workflow assembled from independent components."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from boxmot import create_tracker
from boxmot.detectors import Detector, DetectorSpec, create_detector
from boxmot.detectors.config import resolve_detector_spec
from boxmot.engine.config.trackers import edgetam_checkpoint, resolve_tracker_options
from boxmot.engine.tracking.profiling import RuntimeProfiler, profile_components, startup_stage
from boxmot.engine.tracking.runner import RunSummary, TrackingRunner
from boxmot.engine.tracking.sinks import (
    DisplaySink,
    JsonLinesSink,
    MotSink,
    NullSink,
    SharedRenderingSink,
    TrackSink,
    VideoSink,
)
from boxmot.engine.tracking.sources import FrameSource, create_frame_source
from boxmot.engine.ui.logging import suppress_boxmot_logs
from boxmot.engine.ui.reporters.track import TrackWorkflowReporter
from boxmot.engine.ui.workflow.pipeline import PipelineTracker
from boxmot.pipelines import PipelineOutputs, TrackingPipeline
from boxmot.reid import AppearanceEncoder, ReIDEncoderSpec, create_reid_encoder
from boxmot.reid.config import resolve_reid_spec
from boxmot.segmentors import Segmentor, create_segmentor
from boxmot.segmentors.config import resolve_segmentor_spec
from boxmot.trackers import ReIDConfigurableTracker, Tracker, TrackerSpec
from boxmot.utils.devices import resolve_device

if TYPE_CHECKING:
    import numpy as np

    from boxmot.trackers.common.mask_guidance import MaskGuidance, MaskGuidanceConfig


@dataclass(frozen=True, slots=True)
class TrackRun:
    """Completed live run and any engine-owned output paths."""

    summary: RunSummary
    video_path: Path | None = None
    mot_path: Path | None = None
    json_path: Path | None = None


def _classes(value: object) -> tuple[int, ...] | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        parsed = tuple(sorted({int(token.strip()) for token in value.split(",") if token.strip()}))
    else:
        parsed = tuple(sorted({int(item) for item in value}))  # type: ignore[arg-type]
    if any(item < 0 for item in parsed):
        raise ValueError("classes must contain non-negative integer IDs")
    return parsed


def _component_device(args: Any, authored_device: str) -> str:
    """Resolve one command override or authored component device for live execution."""

    selected = getattr(args, "device", None)
    if selected is None:
        selected = "cpu" if authored_device.strip().lower() == "auto" else authored_device
    return str(resolve_device(selected))


def _detector_spec(args: Any, geometry: str) -> DetectorSpec:
    """Resolve detector options and a validated device before model loading."""

    spec, _ = resolve_detector_spec(args.detector, geometry=geometry)
    options = spec.option_values()
    if getattr(args, "conf", None) is not None:
        options["confidence"] = float(args.conf)
    if getattr(args, "iou", None) is not None:
        options["iou"] = float(args.iou)
    classes = _classes(getattr(args, "classes", None))
    if classes is not None:
        options["classes"] = classes
    image_size = getattr(args, "imgsz", None)
    if image_size is not None:
        if isinstance(image_size, int):
            options["image_size"] = (image_size, image_size)
        else:
            options["image_size"] = tuple(int(value) for value in image_size)
    if getattr(args, "agnostic_nms", False):
        options["agnostic_nms"] = True
    return replace(
        spec,
        device=_component_device(args, spec.device),
        options=tuple(sorted(options.items())),
    )


def _reid_spec(args: Any) -> ReIDEncoderSpec:
    """Resolve a ReID selector with the direct-track runtime controls.

    ``track --device`` historically places detector and ReID inference on the
    same device, while ``--half`` controls ReID precision.  Component profiles
    provide artifact and preprocessing defaults, but must not silently put a
    live run into FP16 when the command reports FP32.
    """

    spec, _ = resolve_reid_spec(args.reid)
    return replace(
        spec,
        device=_component_device(args, spec.device),
        precision="fp16" if bool(getattr(args, "half", False)) else "fp32",
    )


def _tracker_spec(args: Any, geometry: str) -> TrackerSpec:
    """Resolve tracker defaults, an optional runtime config, and CLI overrides."""

    options = resolve_tracker_options(args)
    return TrackerSpec(
        name=str(args.tracker),
        backend=str(getattr(args, "tracker_backend", "python")),
        geometry=geometry,
        per_class=bool(getattr(args, "per_class", False) or options.get("per_class", False)),
        class_ids=_classes(getattr(args, "classes", None)),
        options=tuple(sorted(options.items())),
    )


def _mask_guidance_config(args: Any, tracker_spec: TrackerSpec) -> MaskGuidanceConfig | None:
    """Configure propagation of the frames delivered by the tracking workflow."""
    checkpoint = edgetam_checkpoint(args)
    if checkpoint is None:
        return None
    from boxmot.segmentors.propagation.factory import mask_propagation_device
    from boxmot.segmentors.propagation.weights import resolve_edgetam_artifact
    from boxmot.trackers.common.config import load_tracker_config
    from boxmot.trackers.common.mask_guidance import mask_guidance_config_from_options, validate_mask_guidance_spec

    validate_mask_guidance_spec(tracker_spec)
    config = mask_guidance_config_from_options(
        checkpoint=checkpoint,
        device=str(getattr(args, "device", None) or "cuda"),
        options=load_tracker_config(tracker_spec.name, None, tracker_spec.option_dict),
    )
    checkpoint = resolve_edgetam_artifact(checkpoint)
    device = mask_propagation_device(checkpoint, config.device)
    return replace(config, checkpoint=checkpoint, device=str(resolve_device(device)))


def _share_edgetam_model(
    args: Any, config: MaskGuidanceConfig,
) -> tuple[MaskGuidanceConfig | MaskGuidance, Segmentor | None]:
    """Share matching segmentor/guidance weights within this tracking run only."""
    from boxmot.segmentors.propagation.weights import is_edgetam_tflite_bundle

    if is_edgetam_tflite_bundle(config.checkpoint):
        return config, None

    from boxmot.components.artifacts import sha256_artifact
    from boxmot.segmentors.propagation.edgetam import EdgeTAMMaskPropagator
    from boxmot.segmentors.propagation.model import effective_precision
    from boxmot.trackers.common.mask_guidance import MaskGuidance

    spec, _ = resolve_segmentor_spec(args.segmentor, geometry="aabb")
    spec = replace(spec, device=_component_device(args, spec.device))
    if (
        spec.backend != "edgetam"
        or resolve_device(spec.device) != resolve_device(config.device)
        or spec.precision != effective_precision(config.device)
        or spec.artifact_sha256 != sha256_artifact(config.checkpoint)
    ):
        return config, None
    propagator = EdgeTAMMaskPropagator(
        config.checkpoint, device=config.device, max_objects=config.max_objects,
        prompt_overlap=config.prompt_overlap,
    )
    segmentor = create_segmentor(spec, model=propagator.predictor)
    return MaskGuidance(config, propagator=propagator), segmentor


def _output_directory(args: Any) -> Path:
    project = Path(getattr(args, "project", "runs/track"))
    name = str(getattr(args, "name", "track") or "track")
    output = project / name
    output.mkdir(parents=True, exist_ok=True)
    return output


def _default_sinks(
    args: Any, *, guidance_mask_provider: Callable[[], Mapping[int, np.ndarray]] | None = None
) -> tuple[tuple[TrackSink, ...], Path | None, Path | None, Path | None]:
    """Build output sinks, sharing one optional guidance overlay per frame."""
    sinks: list[TrackSink] = []
    video_path = mot_path = json_path = None
    video_sink: VideoSink | None = None
    display_sink: DisplaySink | None = None
    needs_directory = any(bool(getattr(args, name, False)) for name in ("save", "save_txt", "save_json"))
    output = _output_directory(args) if needs_directory else None
    line_width = int(getattr(args, "line_width", 2) or 2)
    if bool(getattr(args, "save", False)):
        assert output is not None
        video_path = output / "tracks.mp4"
        video_sink = VideoSink(
            video_path,
            fps=float(getattr(args, "fps", None) or 30.0),
            line_width=line_width,
        )
    if bool(getattr(args, "show", False)):
        display_sink = DisplaySink(line_width=line_width)
    if video_sink is not None and display_sink is not None:
        sinks.append(
            SharedRenderingSink(
                (video_sink, display_sink),
                line_width=line_width,
                guidance_mask_provider=guidance_mask_provider,
            )
        )
    elif video_sink is not None:
        sinks.append(
            SharedRenderingSink((video_sink,), line_width=line_width, guidance_mask_provider=guidance_mask_provider)
            if guidance_mask_provider is not None
            else video_sink
        )
    if bool(getattr(args, "save_txt", False)):
        assert output is not None
        mot_path = output / "tracks.txt"
        mot_path.unlink(missing_ok=True)
        sinks.append(MotSink(mot_path))
    if bool(getattr(args, "save_json", False)):
        assert output is not None
        json_path = output / "tracks.jsonl"
        json_path.unlink(missing_ok=True)
        sinks.append(JsonLinesSink(json_path))
    if display_sink is not None and video_sink is None:
        sinks.append(
            SharedRenderingSink((display_sink,), line_width=line_width, guidance_mask_provider=guidance_mask_provider)
            if guidance_mask_provider is not None
            else display_sink
        )
    if not sinks:
        sinks.append(NullSink())
    return tuple(sinks), video_path, mot_path, json_path


def run_track(
    args: Any,
    *,
    detector: Detector | None = None,
    segmentor: Segmentor | None = None,
    encoder: AppearanceEncoder | None = None,
    tracker: Tracker | None = None,
    source: FrameSource | None = None,
    sinks: tuple[TrackSink, ...] | None = None,
    ui_pipeline: PipelineTracker | None = None,
) -> TrackRun:
    """Run one source through the v24 component/pipeline/engine stack."""

    profiler = RuntimeProfiler()
    startup_timings_ms: dict[str, float] = {}
    geometry = str(getattr(args, "geometry", "aabb") or "aabb")
    if geometry not in {"aabb", "obb"}:
        raise ValueError("geometry must be 'aabb' or 'obb'")
    if getattr(args, "device", None) is not None:
        resolve_device(args.device)
    tracker_was_injected = tracker is not None
    tracker_spec = _tracker_spec(args, geometry) if tracker is None else None
    if edgetam_checkpoint(args) is not None and tracker is not None:
        raise ValueError(
            "Configure mask guidance on the supplied tracker instead of combining "
            "--edgetam with an injected tracker."
        )
    mask_guidance = _mask_guidance_config(args, tracker_spec) if tracker_spec is not None else None

    if detector is None:
        if ui_pipeline is not None:
            ui_pipeline.update("Loading detector…")
        with startup_stage(startup_timings_ms, "detector_load"):
            detector = create_detector(_detector_spec(args, geometry))
    if tracker is None:
        if ui_pipeline is not None:
            ui_pipeline.update("Loading tracker…")
        with startup_stage(startup_timings_ms, "tracker_load"):
            if (
                mask_guidance is not None
                and segmentor is None
                and getattr(args, "segmentor", None) is not None
                and not detector.capabilities.provides_masks
            ):
                mask_guidance, segmentor = _share_edgetam_model(args, mask_guidance)
            tracker = (
                create_tracker(tracker_spec, mask_guidance=mask_guidance)
                if mask_guidance is not None
                else create_tracker(tracker_spec)
            )
            requirements = tracker.requirements
            generates_embeddings = getattr(tracker, "generates_embeddings", False)
            if not isinstance(generates_embeddings, bool):
                raise TypeError("tracker.generates_embeddings must be bool.")
            if (
                encoder is None
                and requirements.embeddings
                and generates_embeddings
                and not detector.capabilities.provides_embeddings
                and getattr(args, "reid", None) is not None
            ):
                if not isinstance(tracker, ReIDConfigurableTracker):
                    raise TypeError(
                        f"Tracker {tracker.name!r} declares internal embedding generation "
                        "but does not implement configure_reid()."
                    )
                tracker.configure_reid(_reid_spec(args))

    requirements = tracker.requirements
    generates_embeddings = getattr(tracker, "generates_embeddings", False)
    if not isinstance(generates_embeddings, bool):
        raise TypeError("tracker.generates_embeddings must be bool.")
    if (
        encoder is None
        and requirements.embeddings
        and not detector.capabilities.provides_embeddings
        and (not generates_embeddings or (tracker_was_injected and getattr(args, "reid", None) is not None))
    ):
        reference = getattr(args, "reid", None)
        if reference is None:
            raise ValueError(f"Tracker {tracker.name!r} requires embeddings; configure --reid ID_OR_YAML.")
        if ui_pipeline is not None:
            ui_pipeline.update("Loading appearance encoder…")
        with startup_stage(startup_timings_ms, "reid_load"):
            encoder = create_reid_encoder(_reid_spec(args))

    segmentor_reference = getattr(args, "segmentor", None)
    needs_masks = requirements.masks or bool(encoder is not None and encoder.requirements.masks)
    detector_masks = detector.capabilities.provides_masks
    if segmentor is None and (needs_masks or segmentor_reference is not None) and not detector_masks:
        if segmentor_reference is None:
            raise ValueError(f"Tracker {tracker.name!r} requires masks; configure --segmentor ID_OR_YAML.")
        segmentor_spec, _ = resolve_segmentor_spec(segmentor_reference, geometry=geometry)
        segmentor_spec = replace(segmentor_spec, device=_component_device(args, segmentor_spec.device))
        if ui_pipeline is not None:
            ui_pipeline.update("Loading segmentor…")
        with startup_stage(startup_timings_ms, "segmentor_load"):
            segmentor = create_segmentor(segmentor_spec)

    if source is None:
        if ui_pipeline is not None:
            ui_pipeline.update("Opening frame source…")
        with startup_stage(startup_timings_ms, "source_open"):
            source = create_frame_source(
                getattr(args, "source", "0"),
                stride=int(getattr(args, "vid_stride", 1) or 1),
            )
    if sinks is None:
        with startup_stage(startup_timings_ms, "output_prepare"):
            provider = (
                (lambda owner=tracker: owner.guidance_masks)
                if getattr(tracker, "guidance_masks", None) is not None
                else None
            )
            sinks, video_path, mot_path, json_path = _default_sinks(args, guidance_mask_provider=provider)
    else:
        video_path = mot_path = json_path = None

    with startup_stage(startup_timings_ms, "pipeline_prepare"):
        detector, segmentor, encoder, tracker = profile_components(
            profiler,
            detector=detector,
            segmentor=segmentor,
            reid=encoder,
            tracker=tracker,
        )
        tracking_pipeline = TrackingPipeline(
            detector=detector,
            tracker=tracker,
            segmentor=segmentor,
            reid=encoder,
            outputs=PipelineOutputs(masks=segmentor_reference is not None),
        )
    if ui_pipeline is not None:
        ui_pipeline.advance("Processing frames…")

    def _progress(frame_count: int, frame: Any, _result: Any) -> None:
        if ui_pipeline is not None:
            ui_pipeline.update(f"Processed {frame_count} frame(s) • {frame.sample_id}")

    runner = TrackingRunner(
        source,
        tracking_pipeline,
        sinks=sinks,
        progress=_progress if ui_pipeline is not None else None,
        profiler=profiler,
        startup_timings_ms=startup_timings_ms,
    )
    for _frame, _result in runner.run():
        pass
    assert runner.summary is not None
    return TrackRun(
        summary=runner.summary,
        video_path=video_path,
        mot_path=mot_path,
        json_path=json_path,
    )


def main(args: Any) -> TrackRun:
    """CLI entry point."""

    reporter = TrackWorkflowReporter(args)
    pipeline = reporter.pipeline()
    with pipeline:
        with suppress_boxmot_logs(True, level="WARNING"):
            result = run_track(args, ui_pipeline=pipeline)
        output_dir = next(
            (path.parent for path in (result.video_path, result.mot_path, result.json_path) if path is not None),
            None,
        )
        pipeline.finish(reporter.result(result), exp_dir=output_dir)
        return result


__all__ = ("TrackRun", "main", "run_track")
