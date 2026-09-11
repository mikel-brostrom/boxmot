"""Pipelines forward capture metadata; trackers own timestamp validation/state."""

from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
import pytest
import torch

from boxmot.detectors.protocols import DetectorCapabilities
from boxmot.pipelines import TrackingPipeline
from boxmot.reid.protocols import EncoderRequirements
from boxmot.structures import Boxes, Detections, Frame, Tracks
from boxmot.trackers.bytetrack.tracker import ByteTrack
from boxmot.trackers.common.protocols import TrackerRequirements


def _frame(index: int, timestamp: float | None) -> Frame:
    """Create a uniquely indexed sample from one video sequence."""
    return Frame(
        image=torch.zeros((3, 8, 12), dtype=torch.uint8),
        sample_id=f"sequence/{index}",
        sequence_id="sequence",
        frame_index=index,
        timestamp_s=timestamp,
    )


def _detections(frame: Frame) -> Detections:
    """Return one cheap canonical detection for the requested sample."""
    return Detections(
        geometry=Boxes(torch.tensor([[1.0, 1.0, 4.0, 6.0]])),
        scores=torch.tensor([0.9]),
        class_ids=torch.tensor([0]),
        sample_id=frame.sample_id,
    )


class _Detector:
    capabilities = DetectorCapabilities()

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def predict(self, frames: tuple[Frame, ...]) -> list[Detections]:
        self.events.append("detect")
        return [_detections(frame) for frame in frames]


class _Encoder:
    embedding_dim = 4
    requirements = EncoderRequirements()

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def encode(self, frames: tuple[Frame, ...], detections: tuple[Detections, ...]) -> list[torch.Tensor]:
        self.events.append("encode")
        return [torch.ones((len(item), 4)) for item in detections]


class _Tracker:
    """Record pipeline calls while real ByteTrack owns elapsed-time handling."""

    supports_variable_dt = True
    supports_obb = False
    generates_embeddings = False

    def __init__(self, events: list[str], *, variable_dt: bool = True) -> None:
        self.events = events
        self.requirements = TrackerRequirements(embeddings=True)
        self.received: list[tuple[float | None, Frame | None]] = []
        self.variable_dt = variable_dt
        self.tracker = ByteTrack(variable_dt=variable_dt)

    def validate_timing(self, frame: Frame | None, *, timestamp_s: float | None = None) -> float | None:
        return self.tracker.validate_timing(frame, timestamp_s=timestamp_s)

    def update(self, detections: Detections, frame: Frame | None = None, *, timestamp_s: float | None = None) -> Tracks:
        # This spy consumes appearance to check ordering; ByteTrack consumes boxes.
        assert detections.embeddings is not None
        tracks = self.tracker.update(replace(detections, embeddings=None), frame, timestamp_s=timestamp_s)
        self.events.append("track")
        self.received.append((self.tracker._prediction_dt, frame))
        return tracks

    def reset(self) -> None:
        self.tracker.reset()
        self.events.append("reset")


def _pipeline(*, variable_dt: bool = True) -> tuple[TrackingPipeline, _Tracker, list[str]]:
    """Count perception and tracking calls to detect validation side effects."""
    events: list[str] = []
    tracker = _Tracker(events, variable_dt=variable_dt)
    pipeline = TrackingPipeline(detector=_Detector(events), tracker=tracker, reid=_Encoder(events))
    return pipeline, tracker, events


def _step(pipeline: TrackingPipeline, frame: Frame, *, detect: bool, **kwargs: object) -> None:
    """Exercise detection and caller-supplied-detection entrypoints alike."""
    if detect:
        pipeline.step(frame, **kwargs)
    else:
        pipeline.step_detections(frame, _detections(frame), **kwargs)


@pytest.mark.parametrize("detect", [False, True])
def test_pipeline_forwards_frames_for_tracker_owned_elapsed_time(detect: bool) -> None:
    pipeline, tracker, _ = _pipeline()
    frames = [_frame(index, timestamp) for index, timestamp in enumerate((100.0, 100.1, 100.6))]
    for frame in frames:
        _step(pipeline, frame, detect=detect)
    assert tracker.received[0][0] is None
    assert [entry[0] for entry in tracker.received[1:]] == pytest.approx([0.1, 0.5])
    assert [frame for _, frame in tracker.received] == frames
    assert not hasattr(pipeline, "_last_timestamp_s")


@pytest.mark.parametrize("detect", [False, True])
def test_missing_timestamps_keep_tracker_fixed_step_model(detect: bool) -> None:
    pipeline, tracker, _ = _pipeline(variable_dt=False)
    frames = [_frame(index, None) for index in range(3)]
    for frame in frames:
        _step(pipeline, frame, detect=detect)
    assert [entry[0] for entry in tracker.received] == [None, None, None]
    assert all(frame is None for _, frame in tracker.received)


def test_pipeline_reset_resets_tracker_timestamp_anchor() -> None:
    pipeline, tracker, events = _pipeline()
    _step(pipeline, _frame(0, 100.0), detect=False)
    _step(pipeline, _frame(1, 100.5), detect=False)
    pipeline.reset()
    _step(pipeline, _frame(0, 10.0), detect=False)
    _step(pipeline, _frame(1, 10.25), detect=False)
    assert [entry[0] for entry in tracker.received] == [None, 0.5, None, 0.25]
    assert events.count("reset") == 1


@pytest.mark.parametrize("detect", [False, True])
@pytest.mark.parametrize("timestamp", [None, 10.0, 9.9, np.nan, np.inf])
def test_invalid_timestamp_rejects_frame_before_perception_or_tracking(detect: bool, timestamp: float | None) -> None:
    pipeline, tracker, events = _pipeline()
    _step(pipeline, _frame(0, 10.0), detect=detect)
    invalid = _frame(1, None)
    object.__setattr__(invalid, "timestamp_s", timestamp)
    previous_events = events.copy()
    with pytest.raises(ValueError, match="timestamp|timing"):
        _step(pipeline, invalid, detect=detect)
    assert events == previous_events
    assert len(tracker.received) == 1
    _step(pipeline, _frame(1, 10.25), detect=detect)
    assert tracker.received[-1][0] == pytest.approx(0.25)


@pytest.mark.parametrize("detect", [False, True])
@pytest.mark.parametrize("timestamp", [None, np.nan, np.inf])
def test_missing_or_nonfinite_first_timestamp_rejects_before_perception(detect: bool, timestamp: float | None) -> None:
    pipeline, tracker, events = _pipeline()
    frame = _frame(0, None)
    object.__setattr__(frame, "timestamp_s", timestamp)
    with pytest.raises(ValueError, match="timestamp"):
        _step(pipeline, frame, detect=detect)
    assert events == []
    assert tracker.received == []


@pytest.mark.parametrize("detect", [False, True])
def test_fixed_mode_metadata_changes_do_not_enable_timed_prediction(detect: bool) -> None:
    pipeline, tracker, _ = _pipeline(variable_dt=False)

    for index, timestamp in enumerate((None, 10.0, 9.0, 9.0, None)):
        _step(pipeline, _frame(index, timestamp), detect=detect)

    assert tracker.variable_dt is False
    assert tracker.received == [(None, None)] * 5


@pytest.mark.parametrize("detect", [False, True])
def test_pipeline_exposes_no_elapsed_time_argument(detect: bool) -> None:
    pipeline, tracker, events = _pipeline()
    assert "use_timestamps" not in inspect.signature(TrackingPipeline).parameters
    assert "dt" not in inspect.signature(TrackingPipeline.step).parameters
    assert "dt" not in inspect.signature(TrackingPipeline.step_detections).parameters
    with pytest.raises(TypeError, match="dt"):
        _step(pipeline, _frame(0, 10.0), detect=detect, dt=0.25)
    assert events == []
    assert tracker.received == []


@pytest.mark.parametrize("detect", [False, True])
def test_unsupported_tracker_does_not_receive_unneeded_frame_metadata(detect: bool) -> None:
    pipeline, tracker, _ = _pipeline(variable_dt=False)
    tracker.supports_variable_dt = False
    for index, timestamp in enumerate((10.0, 9.0, 9.0)):
        _step(pipeline, _frame(index, timestamp), detect=detect)
    assert tracker.received == [(None, None), (None, None), (None, None)]


@pytest.mark.parametrize("detect", [False, True])
def test_perception_failure_does_not_commit_tracker_timestamp(detect: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline, tracker, events = _pipeline()
    _step(pipeline, _frame(0, 10.0), detect=detect)
    encoder_type = type(pipeline.reid)
    original_encode = encoder_type.encode

    def fail_encode(*args: object, **kwargs: object) -> list[torch.Tensor]:
        raise RuntimeError("encoder failed")

    monkeypatch.setattr(encoder_type, "encode", fail_encode)
    with pytest.raises(RuntimeError, match="encoder failed"):
        _step(pipeline, _frame(1, 10.5), detect=detect)
    assert len(tracker.received) == 1
    monkeypatch.setattr(encoder_type, "encode", original_encode)
    _step(pipeline, _frame(1, 10.5), detect=detect)
    assert tracker.received[-1][0] == pytest.approx(0.5)
    assert events.count("track") == 2
