from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest
import torch

from boxmot import BotSortConfig, ByteTrackConfig
from boxmot.detectors.protocols import DetectorCapabilities
from boxmot.pipelines import PerceptionPipeline, PipelineOutputs, PipelineResult, TrackingPipeline
from boxmot.reid.protocols import EncoderRequirements
from boxmot.structures import Boxes, Detections, Frame, MaskBatch, OrientedBoxes, Tracks
from boxmot.trackers.common.protocols import TrackerRequirements


def _frame(
    sample_id: str,
    value: int = 0,
    *,
    sequence_id: str | None = None,
    frame_index: int | None = None,
) -> Frame:
    return Frame(
        torch.full((3, 8, 12), value, dtype=torch.uint8),
        sample_id,
        sequence_id=sequence_id,
        frame_index=frame_index,
    )


def _detections(frame: Frame, *, empty: bool = False, masks: bool = False, embeddings: bool = False) -> Detections:
    count = 0 if empty else 1
    geometry = Boxes(
        torch.empty((0, 4), dtype=torch.float32) if empty else torch.tensor([[1.0, 1.0, 4.0, 6.0]], dtype=torch.float32)
    )
    result = Detections(
        geometry=geometry,
        scores=torch.full((count,), 0.9, dtype=torch.float32),
        class_ids=torch.zeros((count,), dtype=torch.int64),
        sample_id=frame.sample_id,
    )
    if masks:
        result = result.with_masks(MaskBatch(torch.ones((count, frame.height, frame.width), dtype=torch.bool)))
    if embeddings:
        result = result.with_embeddings(torch.ones((count, 4), dtype=torch.float32))
    return result


class _Detector:
    def __init__(
        self,
        results,
        events,
        *,
        provides_masks=False,
        provides_embeddings=False,
        supports_aabb=True,
        supports_obb=False,
    ):
        self.results = results
        self.events = events
        self.capabilities = DetectorCapabilities(
            provides_masks=provides_masks,
            provides_embeddings=provides_embeddings,
            supports_aabb=supports_aabb,
            supports_obb=supports_obb,
        )

    def predict(self, frames):
        self.events.append(("detect", tuple(frame.sample_id for frame in frames)))
        return self.results


class _Segmentor:
    def __init__(self, events):
        self.events = events

    def segment(self, frames, detections):
        self.events.append(("segment", tuple(frame.sample_id for frame in frames)))
        return [
            MaskBatch(torch.ones((len(item), frame.height, frame.width), dtype=torch.bool))
            for frame, item in zip(frames, detections)
        ]


class _Encoder:
    embedding_dim = 4

    def __init__(self, events, *, requires_masks=False):
        self.events = events
        self.requirements = EncoderRequirements(masks=requires_masks)

    def encode(self, frames, detections):
        self.events.append(
            (
                "embed",
                tuple(frame.sample_id for frame in frames),
                tuple(item.masks is not None for item in detections),
            )
        )
        return [torch.full((len(item), self.embedding_dim), 2.0, dtype=torch.float32) for item in detections]


class _Tracker:
    name = "fake"
    supports_obb = False

    def __init__(
        self,
        events,
        *,
        embeddings=False,
        masks=False,
        frame=False,
        frame_dimensions_only=False,
        generates_embeddings=False,
    ):
        self.events = events
        self.requirements = TrackerRequirements(
            embeddings=embeddings,
            masks=masks,
            frame=frame,
            frame_dimensions_only=frame_dimensions_only,
        )
        self.generates_embeddings = generates_embeddings
        self.received = None
        self.reset_calls = 0

    def update(self, detections, frame=None):
        self.events.append(("track", detections.sample_id))
        self.received = (detections, frame)
        count = len(detections)
        return Tracks(
            geometry=detections.geometry,
            track_ids=torch.arange(count, dtype=torch.int64),
            scores=detections.scores,
            class_ids=detections.class_ids,
            detection_indices=torch.arange(count, dtype=torch.int64),
            sample_id=detections.sample_id,
            masks=detections.masks,
        )

    def reset(self):
        self.reset_calls += 1


class _OutOfRangeTracker(_Tracker):
    def update(self, detections, frame=None):
        tracks = super().update(detections, frame)
        return Tracks(
            geometry=tracks.geometry,
            track_ids=tracks.track_ids,
            scores=tracks.scores,
            class_ids=tracks.class_ids,
            detection_indices=torch.full((len(tracks),), len(detections), dtype=torch.int64),
            sample_id=tracks.sample_id,
            masks=tracks.masks,
        )


def test_tracking_pipeline_rejects_out_of_range_detection_indices() -> None:
    events = []
    frame = _frame("one")
    pipeline = TrackingPipeline(detector=None, tracker=_OutOfRangeTracker(events))

    with pytest.raises(ValueError, match="outside the current detection batch"):
        pipeline.step_detections(frame, _detections(frame))


def test_perception_runs_only_requested_stages_in_order_and_skips_empty_model_work() -> None:
    events = []
    first, second = _frame("first"), _frame("second")
    detector = _Detector([_detections(first), _detections(second, empty=True)], events)
    pipeline = PerceptionPipeline(
        detector=detector,
        segmentor=_Segmentor(events),
        reid=_Encoder(events, requires_masks=True),
        outputs=PipelineOutputs(embeddings=True),
    )

    results = pipeline.process((first, second))

    assert events == [
        ("detect", ("first", "second")),
        ("segment", ("first",)),
        ("embed", ("first",), (True,)),
    ]
    assert results[0].masks is not None and results[0].embeddings is not None
    assert results[1].masks is not None and results[1].masks.values.shape == (0, 8, 12)
    assert results[1].embeddings is not None and results[1].embeddings.shape == (0, 4)


def test_perception_does_not_run_configured_but_unrequested_components() -> None:
    events = []
    frame = _frame("one")
    pipeline = PerceptionPipeline(
        detector=_Detector([_detections(frame)], events),
        segmentor=_Segmentor(events),
        reid=_Encoder(events),
    )

    result = pipeline.process([frame])[0]

    assert events == [("detect", ("one",))]
    assert result.masks is None
    assert result.embeddings is None


def test_perception_enriches_caller_detections_without_running_detector() -> None:
    events = []
    frame = _frame("one")
    pipeline = PerceptionPipeline(
        detector=_Detector([], events),
        segmentor=_Segmentor(events),
        reid=_Encoder(events),
    )

    result = pipeline.enrich(
        [frame],
        [_detections(frame)],
        TrackerRequirements(masks=True, embeddings=True),
    )[0]

    assert events == [("segment", ("one",)), ("embed", ("one",), (True,))]
    assert result.masks is not None
    assert result.embeddings is not None


def test_perception_enrich_supports_no_detector_but_process_requires_one() -> None:
    frame = _frame("one")
    detections = _detections(frame)
    pipeline = PerceptionPipeline(detector=None)

    assert pipeline.enrich([frame], [detections]) == [detections]
    with pytest.raises(RuntimeError, match=r"requires a detector; use enrich\(\)"):
        pipeline.process([frame])


@pytest.mark.parametrize("detect", (False, True))
def test_perception_binds_missing_sample_ids_before_enrichment(detect: bool) -> None:
    """Frame pairing supplies identity without mutating caller or detector results."""
    events = []
    frames = [_frame("first"), _frame("second")]
    detections = [dataclasses.replace(_detections(frame), sample_id=None) for frame in frames]

    class CheckingEncoder(_Encoder):
        def encode(self, frames: list[Frame], detections: list[Detections]) -> list[torch.Tensor]:
            assert [item.sample_id for item in detections] == [frame.sample_id for frame in frames]
            return super().encode(frames, detections)

    pipeline = PerceptionPipeline(
        detector=_Detector(detections, events) if detect else None,
        reid=CheckingEncoder(events),
        outputs=PipelineOutputs(embeddings=True),
    )

    results = pipeline.process(frames) if detect else pipeline.enrich(frames, detections)

    assert [item.sample_id for item in results] == ["first", "second"]
    assert all(item.embeddings is not None for item in results)
    assert all(item.sample_id is None and item.embeddings is None for item in detections)
    assert all(result.geometry is original.geometry for result, original in zip(results, detections))


@pytest.mark.parametrize("detect", (False, True))
def test_perception_rejects_explicit_sample_mismatch_before_enrichment(detect: bool) -> None:
    """Optional identity must not disable checks on supplied detector identities."""
    frame = _frame("one")
    detections = [dataclasses.replace(_detections(frame), sample_id="other")]
    events = []
    pipeline = PerceptionPipeline(
        detector=_Detector(detections, events) if detect else None,
        reid=_Encoder(events),
        outputs=PipelineOutputs(embeddings=True),
    )

    with pytest.raises(ValueError, match="sample_id"):
        if detect:
            pipeline.process([frame])
        else:
            pipeline.enrich([frame], detections)

    assert all(event[0] != "embed" for event in events)


def test_perception_preserves_detector_enrichments_and_skips_other_providers() -> None:
    events = []
    frame = _frame("one")
    detection = _detections(frame, masks=True, embeddings=True)
    pipeline = PerceptionPipeline(
        detector=_Detector(
            [detection],
            events,
            provides_masks=True,
            provides_embeddings=True,
        ),
        outputs=PipelineOutputs(masks=True, embeddings=True),
    )

    assert pipeline.process([frame]) == [detection]
    assert events == [("detect", ("one",))]


def test_perception_rejects_missing_provider_and_misaligned_detector_batch() -> None:
    frame = _frame("one")
    with pytest.raises(ValueError, match="neither the detector nor a segmentor"):
        PerceptionPipeline(_Detector([_detections(frame)], []), outputs=PipelineOutputs(masks=True))
    with pytest.raises(ValueError, match="neither the detector nor a ReID"):
        PerceptionPipeline(_Detector([_detections(frame)], []), outputs=PipelineOutputs(embeddings=True))

    pipeline = PerceptionPipeline(_Detector([], []))
    with pytest.raises(ValueError, match="0 results for 1 frames"):
        pipeline.process([frame])


def test_perception_rejects_detector_geometry_not_declared_by_capabilities() -> None:
    frame = _frame("one")
    detections = Detections(
        geometry=OrientedBoxes(torch.tensor([[4.0, 3.0, 4.0, 2.0, 0.0]], dtype=torch.float32)),
        scores=torch.tensor([0.9], dtype=torch.float32),
        class_ids=torch.tensor([0], dtype=torch.int64),
        sample_id=frame.sample_id,
    )
    pipeline = PerceptionPipeline(_Detector([detections], []))

    with pytest.raises(ValueError, match="OBB geometry"):
        pipeline.process([frame])


def test_tracking_unions_requirements_with_outputs_and_returns_exact_result() -> None:
    events = []
    frame = _frame("one")
    tracker = _Tracker(events, embeddings=True, masks=False, frame=True)
    pipeline = TrackingPipeline(
        detector=_Detector([_detections(frame)], events),
        segmentor=_Segmentor(events),
        reid=_Encoder(events, requires_masks=True),
        tracker=tracker,
        outputs=PipelineOutputs(masks=True),
    )

    result = pipeline.step(frame)

    assert isinstance(result, PipelineResult)
    assert tuple(field.name for field in dataclasses.fields(result)) == ("detections", "tracks")
    assert events == [
        ("detect", ("one",)),
        ("segment", ("one",)),
        ("embed", ("one",), (True,)),
        ("track", "one"),
    ]
    tracker_detections, tracker_frame = tracker.received
    assert tracker_frame is frame
    assert tracker_detections.masks is None
    assert tracker_detections.embeddings is result.detections.embeddings
    assert result.detections.masks is not None
    assert result.detections.embeddings is not None
    assert result.tracks.sample_id == frame.sample_id


def test_output_enrichments_are_preserved_without_passing_unused_inputs_to_tracker() -> None:
    """A box tracker can score masks while retaining separately requested features."""
    from boxmot.trackers.bytetrack.tracker import ByteTrack

    events = []
    frame = _frame("one")
    tracker = ByteTrack(
        config=ByteTrackConfig(min_hits=1),
    )
    pipeline = TrackingPipeline(
        detector=_Detector([_detections(frame)], events),
        segmentor=_Segmentor(events),
        reid=_Encoder(events),
        tracker=tracker,
        outputs=PipelineOutputs(masks=True, embeddings=True),
    )

    result = pipeline.step(frame)

    assert len(result.tracks) == 1
    assert result.detections.masks is not None
    assert result.detections.embeddings is not None
    assert result.tracks.masks is None
    with pytest.raises(ValueError, match="does not use detection embeddings"):
        tracker.update(result.detections)


@pytest.mark.parametrize("cached_embeddings", (False, True))
@pytest.mark.parametrize("detect", (False, True))
@pytest.mark.parametrize("reid_source", ("spec", "config", "encoder"))
@pytest.mark.parametrize("use_factory", (False, True))
def test_live_reid_mask_requirements_apply_only_when_features_are_missing(
    monkeypatch: pytest.MonkeyPatch, cached_embeddings: bool, detect: bool, reid_source: str, use_factory: bool
) -> None:
    """Mask-aware live ReID gets segmentation without requiring it for cached features."""
    from boxmot import ReIDConfig, create_tracker
    from boxmot.reid import factory as reid_factory
    from boxmot.reid.specs import ReIDEncoderSpec
    from boxmot.trackers.botsort.tracker import BotSort

    events = []
    frame = _frame("one")
    detections = _detections(frame, embeddings=cached_embeddings)
    encoder = _Encoder(events, requires_masks=True)
    constructed = []

    def create_encoder(spec: ReIDConfig | ReIDEncoderSpec) -> _Encoder:
        constructed.append(spec)
        return encoder

    monkeypatch.setattr(reid_factory, "create_reid_encoder", create_encoder)
    config = ReIDConfig(model="unused.onnx", allow_download=False)
    selected_reid = {"spec": None, "config": config, "encoder": encoder}[reid_source]
    tracker = (
        create_tracker("botsort", use_cmc=False, reid=selected_reid)
        if use_factory
        else BotSort(config=BotSortConfig(use_cmc=False), reid=selected_reid)
    )
    if reid_source == "spec":
        tracker.configure_reid(ReIDEncoderSpec("onnx", artifact="unused.onnx"))
    pipeline = TrackingPipeline(
        detector=_Detector([detections], events, provides_embeddings=cached_embeddings) if detect else None,
        tracker=tracker,
        segmentor=_Segmentor(events),
    )
    assert not constructed
    result = pipeline.step(frame) if detect else pipeline.step_detections(frame, detections)

    assert len(result.tracks) == 1
    assert bool(constructed) is (not cached_embeddings and reid_source != "encoder")
    if constructed and reid_source == "config":
        assert constructed == [config]
    assert (result.detections.masks is not None) is (not cached_embeddings)
    assert any(event[0] == "embed" and event[2] == (True,) for event in events) is (not cached_embeddings)


def test_prebuilt_encoder_is_reusable_by_pipeline_and_tracker_without_duplicate_inference() -> None:
    """The same canonical component may own inference in either composition."""
    from boxmot import BotSort

    events = []
    encoder = _Encoder(events)
    frame = _frame("one")
    tracker = BotSort(config=BotSortConfig(use_cmc=False), reid=encoder)
    pipeline = TrackingPipeline(detector=None, tracker=tracker, reid=encoder)
    result = pipeline.step_detections(frame, _detections(frame))
    assert len(result.tracks) == 1
    assert [event[0] for event in events] == ["embed"]
    assert result.detections.embeddings is not None

    # Reset changes sequence state, not ownership of the supplied encoder.
    pipeline.reset()
    tracker.update(_detections(frame), frame)
    assert [event[0] for event in events] == ["embed", "embed"]


def test_tracking_passes_no_frame_when_tracker_does_not_require_it() -> None:
    events = []
    frame = _frame("one")
    tracker = _Tracker(events)
    result = TrackingPipeline(detector=_Detector([_detections(frame)], events), tracker=tracker).step(frame)
    assert tracker.received == (result.detections, None)


def test_tracking_passes_frame_context_to_dimensions_only_tracker() -> None:
    events = []
    frame = _frame("one")
    tracker = _Tracker(events, frame=True, frame_dimensions_only=True)

    result = TrackingPipeline(detector=_Detector([_detections(frame)], events), tracker=tracker).step(frame)

    assert tracker.received == (result.detections, frame)


def test_tracking_allows_internal_live_embeddings_and_passes_the_frame() -> None:
    events = []
    frame = _frame("one")
    tracker = _Tracker(events, embeddings=True, generates_embeddings=True)
    pipeline = TrackingPipeline(
        detector=_Detector([_detections(frame)], events),
        tracker=tracker,
    )

    result = pipeline.step(frame)

    assert events == [("detect", ("one",)), ("track", "one")]
    assert result.detections.embeddings is None
    assert tracker.received == (result.detections, frame)


@pytest.mark.parametrize("empty", (False, True))
def test_tracking_step_detections_allows_internal_live_embeddings(empty: bool) -> None:
    events = []
    frame = _frame("one")
    tracker = _Tracker(events, embeddings=True, generates_embeddings=True)
    pipeline = TrackingPipeline(detector=None, tracker=tracker)

    result = pipeline.step_detections(frame, _detections(frame, empty=empty))

    assert events == [("track", "one")]
    assert result.detections.embeddings is None
    assert tracker.received == (result.detections, None if empty else frame)


def test_tracking_precomputed_embeddings_bypass_internal_frame_need() -> None:
    events = []
    frame = _frame("one")
    tracker = _Tracker(events, embeddings=True, generates_embeddings=True)
    pipeline = TrackingPipeline(
        detector=_Detector(
            [_detections(frame, embeddings=True)],
            events,
            provides_embeddings=True,
        ),
        tracker=tracker,
    )

    result = pipeline.step(frame)

    assert result.detections.embeddings is not None
    assert tracker.received == (result.detections, None)


def test_tracking_prefers_explicit_external_reid_to_internal_generation() -> None:
    events = []
    frame = _frame("one")
    tracker = _Tracker(events, embeddings=True, generates_embeddings=True)
    pipeline = TrackingPipeline(
        detector=_Detector([_detections(frame)], events),
        tracker=tracker,
        reid=_Encoder(events),
    )

    result = pipeline.step(frame)

    assert events == [
        ("detect", ("one",)),
        ("embed", ("one",), (False,)),
        ("track", "one"),
    ]
    assert result.detections.embeddings is not None
    assert tracker.received == (result.detections, None)


def test_tracking_requested_embedding_outputs_still_require_an_upstream_provider() -> None:
    frame = _frame("one")
    tracker = _Tracker([], embeddings=True, generates_embeddings=True)

    with pytest.raises(ValueError, match="requires appearance embeddings"):
        TrackingPipeline(
            detector=_Detector([_detections(frame)], []),
            tracker=tracker,
            outputs=PipelineOutputs(embeddings=True),
        )


def test_tracking_step_detections_skips_detector_and_enriches_for_tracker() -> None:
    events = []
    frame = _frame("one")
    tracker = _Tracker(events, embeddings=True)
    pipeline = TrackingPipeline(
        detector=None,
        tracker=tracker,
        reid=_Encoder(events),
    )

    result = pipeline.step_detections(frame, _detections(frame))

    assert events == [("embed", ("one",), (False,)), ("track", "one")]
    assert result.detections.embeddings is not None


@pytest.mark.parametrize("empty", (False, True))
@pytest.mark.parametrize("enrich", (False, True))
def test_tracking_step_detections_binds_missing_sample_id(empty: bool, enrich: bool) -> None:
    """Both cached and enriched paths supply frame identity before tracking."""
    frame = _frame("one")
    detections = dataclasses.replace(_detections(frame, empty=empty), sample_id=None)
    events = []
    tracker = _Tracker(events, embeddings=enrich)
    pipeline = TrackingPipeline(detector=None, tracker=tracker, reid=_Encoder(events) if enrich else None)

    result = pipeline.step_detections(frame, detections)

    assert result.detections.sample_id == result.tracks.sample_id == frame.sample_id
    assert tracker.received == (result.detections, None)
    assert result.detections.geometry is detections.geometry
    assert detections.sample_id is None
    assert detections.embeddings is None


@pytest.mark.parametrize("empty", (False, True))
@pytest.mark.parametrize("payloads", (False, True))
def test_tracking_complete_cached_payloads_validate_once_without_perception(
    monkeypatch: pytest.MonkeyPatch, empty: bool, payloads: bool
) -> None:
    """Cached frames keep one pipeline input check and the public tracker call."""
    events = []
    frame = _frame("one")
    detections = _detections(frame, empty=empty, masks=payloads, embeddings=payloads)
    tracker = _Tracker(events, masks=payloads, embeddings=payloads)
    pipeline = TrackingPipeline(detector=None, tracker=tracker)
    validated = []
    original_validate = Detections.validate

    def validate(item: Detections) -> None:
        validated.append(item)
        original_validate(item)

    def unexpected_enrichment(*args: object, **kwargs: object) -> list[Detections]:
        pytest.fail("Complete cached detections must not enter perception enrichment.")

    monkeypatch.setattr(Detections, "validate", validate)
    monkeypatch.setattr(PerceptionPipeline, "enrich", unexpected_enrichment)

    result = pipeline.step_detections(frame, detections)

    assert validated == [detections]
    assert tracker.received == (detections, None)
    assert events == [("track", "one")]
    assert result.detections is detections


@pytest.mark.parametrize("payloads", (False, True))
def test_perception_complete_cached_payloads_validate_once(
    monkeypatch: pytest.MonkeyPatch, payloads: bool
) -> None:
    """A no-op enrichment checks mutable input without repeating that check."""
    frame = _frame("one")
    detections = _detections(frame, masks=payloads, embeddings=payloads)
    pipeline = PerceptionPipeline(outputs=PipelineOutputs(masks=payloads, embeddings=payloads))
    validated = []
    original_validate = Detections.validate

    def validate(item: Detections) -> None:
        validated.append(item)
        original_validate(item)

    monkeypatch.setattr(Detections, "validate", validate)

    assert pipeline.enrich([frame], [detections]) == [detections]
    assert validated == [detections]


@pytest.mark.parametrize("entrypoint", ("tracking", "perception"))
@pytest.mark.parametrize("field", ("geometry", "scores", "embeddings"))
def test_cached_payloads_are_revalidated_after_caller_mutation(entrypoint: str, field: str) -> None:
    """Frozen containers cannot make their underlying tensor storage trusted."""
    events = []
    frame = _frame("one")
    detections = _detections(frame, embeddings=True)
    tracker = _Tracker(events)
    tracking = TrackingPipeline(detector=None, tracker=tracker)
    perception = PerceptionPipeline()

    def advance() -> object:
        if entrypoint == "tracking":
            return tracking.step_detections(frame, detections)
        return perception.enrich([frame], [detections])

    advance()
    previous_events = events.copy()
    values = detections.geometry.values if field == "geometry" else getattr(detections, field)
    values.fill_(float("nan"))

    with pytest.raises(ValueError, match="finite"):
        advance()
    assert events == previous_events


@pytest.mark.parametrize("mismatch", ("sample", "mask_size"))
def test_tracking_cached_payloads_reject_frame_mismatch_before_tracker(mismatch: str) -> None:
    frame = _frame("one")
    events = []
    tracker = _Tracker(events, masks=True)
    detections = _detections(frame, masks=True)
    if mismatch == "sample":
        detections = dataclasses.replace(detections, sample_id="other")
    else:
        detections = detections.with_masks(MaskBatch(torch.ones((1, 4, 4), dtype=torch.bool)))
    pipeline = TrackingPipeline(detector=None, tracker=tracker)

    with pytest.raises(ValueError, match="sample_id|mask size"):
        pipeline.step_detections(frame, detections)
    assert events == []


def test_tracking_cached_path_validates_mutated_custom_tracker_output() -> None:
    """Custom trackers may modify a Tracks tensor after its constructor ran."""
    class InvalidTracker(_Tracker):
        def update(self, detections: Detections, frame: Frame | None = None) -> Tracks:
            tracks = super().update(detections, frame)
            tracks.track_ids.fill_(-1)
            return tracks

    frame = _frame("one")
    pipeline = TrackingPipeline(detector=None, tracker=InvalidTracker([]))

    with pytest.raises(ValueError, match="track_ids must be non-negative"):
        pipeline.step_detections(frame, _detections(frame))


def test_tracking_cached_embeddings_still_enrich_missing_encoder_masks() -> None:
    """The cached branch honors all configured perception requirements."""
    events = []
    frame = _frame("one")
    tracker = _Tracker(events, embeddings=True)
    pipeline = TrackingPipeline(
        detector=None,
        tracker=tracker,
        segmentor=_Segmentor(events),
        reid=_Encoder(events, requires_masks=True),
    )

    result = pipeline.step_detections(frame, _detections(frame, embeddings=True))

    assert events == [("segment", ("one",)), ("track", "one")]
    assert result.detections.masks is not None


def test_perception_enrichment_rechecks_input_mutated_by_provider() -> None:
    """External providers must not leave invalid data behind the input check."""
    first, second = _frame("first"), _frame("second")
    complete = _detections(first, embeddings=True)

    class MutatingEncoder(_Encoder):
        def encode(self, frames: list[Frame], detections: list[Detections]) -> list[torch.Tensor]:
            # This already-complete object is not reconstructed by enrichment.
            complete.scores.fill_(float("nan"))
            return super().encode(frames, detections)

    pipeline = PerceptionPipeline(reid=MutatingEncoder([]), outputs=PipelineOutputs(embeddings=True))

    with pytest.raises(ValueError, match="finite"):
        pipeline.enrich([first, second], [complete, _detections(second)])


def test_tracking_without_detector_rejects_live_step_with_actionable_error() -> None:
    frame = _frame("one")
    pipeline = TrackingPipeline(detector=None, tracker=_Tracker([]))

    with pytest.raises(RuntimeError, match=r"requires a detector; use step_detections\(\)"):
        pipeline.step(frame)


def test_tracking_enforces_one_increasing_sequence_until_reset() -> None:
    events = []
    tracker = _Tracker(events)
    pipeline = TrackingPipeline(detector=_Detector([], events), tracker=tracker)
    first = _frame("a-2", sequence_id="a", frame_index=2)
    pipeline.step_detections(first, _detections(first))

    repeated = _frame("a-2-again", sequence_id="a", frame_index=2)
    with pytest.raises(ValueError, match="must increase"):
        pipeline.step_detections(repeated, _detections(repeated))
    other = _frame("b-3", sequence_id="b", frame_index=3)
    with pytest.raises(ValueError, match="reset it"):
        pipeline.step_detections(other, _detections(other))
    assert events == [("track", "a-2")]

    pipeline.reset()
    pipeline.step_detections(other, _detections(other))
    assert tracker.reset_calls == 1
    assert events == [("track", "a-2"), ("track", "b-3")]


def test_tracking_fails_fast_when_tracker_requirements_cannot_be_met() -> None:
    frame = _frame("one")
    detector = _Detector([_detections(frame)], [])
    with pytest.raises(ValueError, match="requires appearance embeddings"):
        TrackingPipeline(detector=detector, tracker=_Tracker([], embeddings=True))
    with pytest.raises(ValueError, match="requires masks"):
        TrackingPipeline(detector=detector, tracker=_Tracker([], masks=True))


def test_tracking_validates_declared_payloads_at_runtime() -> None:
    frame = _frame("one")
    missing_embeddings = _Detector([_detections(frame)], [], provides_embeddings=True)
    with pytest.raises(ValueError, match="required embeddings"):
        TrackingPipeline(detector=missing_embeddings, tracker=_Tracker([], embeddings=True)).step(frame)

    missing_masks = _Detector([_detections(frame)], [], provides_masks=True)
    with pytest.raises(ValueError, match="required masks"):
        TrackingPipeline(detector=missing_masks, tracker=_Tracker([], masks=True)).step(frame)


@pytest.mark.parametrize(
    ("values", "message"),
    (
        (torch.ones((1, 4), dtype=torch.float64), "dtype torch.float32"),
        (torch.tensor([[float("nan"), 0.0, 0.0, 0.0]], dtype=torch.float32), "finite"),
    ),
)
def test_perception_validates_reid_embedding_tensor_invariants(values, message) -> None:
    class InvalidEncoder(_Encoder):
        def encode(self, frames, detections):
            return [values]

    frame = _frame("one")
    pipeline = PerceptionPipeline(
        detector=_Detector([_detections(frame)], []),
        reid=InvalidEncoder([]),
        outputs=PipelineOutputs(embeddings=True),
    )
    with pytest.raises((TypeError, ValueError), match=message):
        pipeline.process([frame])


def test_tracking_rejects_geometry_mode_changed_by_tracker() -> None:
    class WrongGeometryTracker(_Tracker):
        supports_obb = True

        def update(self, detections, frame=None):
            count = len(detections)
            return Tracks(
                geometry=OrientedBoxes(torch.tensor([[2.0, 3.0, 3.0, 5.0, 0.0]], dtype=torch.float32)),
                track_ids=torch.arange(count, dtype=torch.int64),
                scores=detections.scores,
                class_ids=detections.class_ids,
                detection_indices=torch.arange(count, dtype=torch.int64),
                sample_id=detections.sample_id,
            )

    frame = _frame("one")
    pipeline = TrackingPipeline(_Detector([], []), WrongGeometryTracker([]))
    with pytest.raises(ValueError, match="OBB geometry for AABB detections"):
        pipeline.step_detections(frame, _detections(frame))


def test_pipeline_outputs_and_result_are_frozen_and_slotted() -> None:
    outputs = PipelineOutputs()
    assert not hasattr(outputs, "__dict__")
    with pytest.raises(dataclasses.FrozenInstanceError):
        outputs.masks = True

    frame = _frame("one")
    result = TrackingPipeline(_Detector([_detections(frame)], []), _Tracker([])).step(frame)
    assert not hasattr(result, "__dict__")
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.tracks = result.tracks


def test_pipeline_modules_do_not_import_engine_or_own_application_io() -> None:
    pipeline_root = Path(__file__).resolve().parents[3] / "boxmot" / "pipelines"
    forbidden_modules = {"boxmot.engine", "cv2"}
    forbidden_names = {"argparse", "click", "pathlib"}
    for source_path in pipeline_root.glob("*.py"):
        module = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(module):
            if isinstance(node, ast.Import):
                imported = {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported = {node.module or ""}
            else:
                continue
            assert not any(
                any(name == forbidden or name.startswith(f"{forbidden}.") for forbidden in forbidden_modules)
                for name in imported
            )
            assert not imported & forbidden_names
