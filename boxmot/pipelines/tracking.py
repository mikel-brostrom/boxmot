"""Single-frame composition of perception components and a stateful tracker."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from boxmot.detectors.protocols import Detector
from boxmot.pipelines.perception import (
    PerceptionPipeline,
    PipelineOutputs,
    _declares_output,
    _encoder_requirements,
)
from boxmot.reid.protocols import AppearanceEncoder
from boxmot.segmentors.protocols import Segmentor
from boxmot.structures import Detections, Frame, Tracks
from boxmot.trackers.common.protocols import Tracker, TrackerRequirements


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """The enriched detections and tracks produced for one frame."""

    detections: Detections
    tracks: Tracks


def _validated_requirements(tracker: Tracker) -> TrackerRequirements:
    requirements = getattr(tracker, "requirements", None)
    if not isinstance(requirements, TrackerRequirements):
        raise TypeError("tracker.requirements must be a TrackerRequirements object.")
    if requirements.detections_3d or requirements.camera:
        raise ValueError(
            "TrackingPipeline cannot supply 3D detections or a CameraModel; "
            "use the tracker Python update() API with those inputs."
        )
    if not callable(getattr(tracker, "update", None)):
        raise TypeError("tracker must implement update().")
    if not callable(getattr(tracker, "reset", None)):
        raise TypeError("tracker must implement reset().")
    if not isinstance(getattr(tracker, "supports_obb", None), bool):
        raise TypeError("tracker.supports_obb must be bool.")
    return requirements


@dataclass(slots=True)
class TrackingPipeline:
    """Run perception and tracking for one caller-owned frame at a time."""

    detector: Detector | None
    tracker: Tracker
    segmentor: Segmentor | None = None
    reid: AppearanceEncoder | None = None
    outputs: PipelineOutputs = PipelineOutputs()
    _perception: PerceptionPipeline = field(init=False, repr=False)
    _requirements: TrackerRequirements = field(init=False, repr=False)
    _perception_requirements: TrackerRequirements = field(init=False, repr=False)
    _generates_embeddings: bool = field(init=False, repr=False)
    _sequence_id: str | None = field(init=False, default=None, repr=False)
    _sequence_is_set: bool = field(init=False, default=False, repr=False)
    _last_frame_index: int | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        requirements = _validated_requirements(self.tracker)
        generates_embeddings = getattr(self.tracker, "generates_embeddings", False)
        if not isinstance(generates_embeddings, bool):
            raise TypeError("tracker.generates_embeddings must be bool.")
        upstream_embeddings = requirements.embeddings and (not generates_embeddings or self.reid is not None)
        effective = PipelineOutputs(
            masks=self.outputs.masks or requirements.masks,
            embeddings=self.outputs.embeddings or upstream_embeddings,
        )
        if (
            self.detector is not None
            and effective.embeddings
            and self.reid is None
            and not _declares_output(self.detector, "embeddings")
        ):
            raise ValueError(
                "This tracker configuration requires appearance embeddings, but no ReID encoder was provided."
            )
        encoder_needs_masks = effective.embeddings and self.reid is not None and _encoder_requirements(self.reid).masks
        if (
            self.detector is not None
            and (effective.masks or encoder_needs_masks)
            and self.segmentor is None
            and not _declares_output(self.detector, "masks")
        ):
            raise ValueError("This pipeline requires masks, but no segmentor was provided.")
        self._requirements = requirements
        self._perception_requirements = TrackerRequirements(
            embeddings=upstream_embeddings,
            masks=requirements.masks,
            frame=requirements.frame,
            frame_dimensions_only=requirements.frame_dimensions_only,
        )
        self._generates_embeddings = generates_embeddings
        self._perception = PerceptionPipeline(
            detector=self.detector,
            segmentor=self.segmentor,
            reid=self.reid,
            outputs=effective,
        )

    def _validate_frame_order(self, frame: Frame) -> None:
        if self._sequence_is_set and frame.sequence_id != self._sequence_id:
            raise ValueError(
                f"TrackingPipeline is bound to sequence {self._sequence_id!r}; "
                f"reset it before processing sequence {frame.sequence_id!r}."
            )
        if (
            frame.frame_index is not None
            and self._last_frame_index is not None
            and frame.frame_index <= self._last_frame_index
        ):
            raise ValueError(
                f"frame_index must increase within a sequence; received {frame.frame_index} "
                f"after {self._last_frame_index}."
            )

    def _track(self, frame: Frame, detections: Detections) -> PipelineResult:
        self._validate_frame_order(frame)
        embeddings_missing = self._requirements.embeddings and detections.embeddings is None
        needs_live_embedding_pixels = embeddings_missing and len(detections) > 0
        live_encoder_masks = needs_live_embedding_pixels and getattr(self.tracker, "reid_requires_masks", False)
        if live_encoder_masks and detections.masks is None:
            detections = self._perception.enrich((frame,), (detections,), TrackerRequirements(masks=True))[0]
        if embeddings_missing and not self._generates_embeddings:
            raise ValueError("Tracker-required embeddings are missing after perception enrichment.")
        if self._requirements.masks and detections.masks is None:
            raise ValueError("Tracker-required masks are missing after perception enrichment.")
        if detections.is_obb and not self.tracker.supports_obb:
            raise ValueError("Tracker does not support oriented detections.")
        if detections.masks is not None and detections.masks.image_size != frame.image_size:
            raise ValueError(
                f"Detection mask size {detections.masks.image_size} does not match frame size {frame.image_size}."
            )
        needs_frame = (
            self._requirements.frame or needs_live_embedding_pixels or getattr(self.tracker, "variable_dt", False)
        )
        # Perception enrichments requested for output/scoring stay in the result;
        # the tracker receives only channels its configured algorithm consumes.
        tracker_masks = getattr(
            getattr(self.tracker, "capabilities", None), "accepts_masks", self._requirements.masks
        ) or live_encoder_masks
        tracker_detections = detections
        if (detections.masks is not None and not tracker_masks) or (
            detections.embeddings is not None and not self._requirements.embeddings
        ):
            tracker_detections = replace(
                detections,
                masks=detections.masks if tracker_masks else None,
                embeddings=detections.embeddings if self._requirements.embeddings else None,
            )
        tracks = self.tracker.update(detections=tracker_detections, frame=frame if needs_frame else None)
        if not isinstance(tracks, Tracks):
            raise TypeError("Tracker.update() must return a Tracks object.")
        tracks.validate()
        if tracks.sample_id != frame.sample_id:
            raise ValueError(f"Tracker result has sample_id {tracks.sample_id!r}; expected {frame.sample_id!r}.")
        if tracks.is_obb != detections.is_obb:
            expected = "OBB" if detections.is_obb else "AABB"
            actual = "OBB" if tracks.is_obb else "AABB"
            raise ValueError(f"Tracker returned {actual} geometry for {expected} detections.")
        if tracks.detection_indices.numel() and bool((tracks.detection_indices >= len(detections)).any()):
            raise ValueError("Tracker returned a detection index outside the current detection batch.")
        self._sequence_id = frame.sequence_id
        self._sequence_is_set = True
        if frame.frame_index is not None:
            self._last_frame_index = frame.frame_index
        return PipelineResult(detections=detections, tracks=tracks)

    def step_detections(self, frame: Frame, detections: Detections) -> PipelineResult:
        """Advance tracking from caller-supplied detections after required enrichment."""
        if not isinstance(frame, Frame):
            raise TypeError(f"frame must be a Frame, not {type(frame).__name__}.")
        if not isinstance(detections, Detections):
            raise TypeError(f"detections must be a Detections object, not {type(detections).__name__}.")
        self._validate_frame_order(frame)
        if getattr(self.tracker, "variable_dt", False):
            self.tracker.validate_timing(frame)
        enriched = self._perception.enrich((frame,), (detections,), self._perception_requirements)[0]
        return self._track(frame, enriched)

    def step(self, frame: Frame) -> PipelineResult:
        """Detect objects and advance the pipeline by one frame."""
        if not isinstance(frame, Frame):
            raise TypeError(f"frame must be a Frame, not {type(frame).__name__}.")
        if self.detector is None:
            raise RuntimeError(
                "TrackingPipeline.step() requires a detector; use step_detections() for caller-supplied detections."
            )
        self._validate_frame_order(frame)
        if getattr(self.tracker, "variable_dt", False):
            self.tracker.validate_timing(frame)
        detections = self._perception.process((frame,))[0]
        return self._track(frame, detections)

    def reset(self) -> None:
        """Reset tracker and sequence-order state for a new stream."""
        self.tracker.reset()
        self._sequence_id = None
        self._sequence_is_set = False
        self._last_frame_index = None


__all__ = ("PipelineResult", "TrackingPipeline")
