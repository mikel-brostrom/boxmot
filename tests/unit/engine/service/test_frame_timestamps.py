"""HTTP capture timestamps affect prediction only with explicit process opt-in."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from boxmot import ByteTrack
from boxmot.engine.service.app import create_app
from boxmot.engine.service.models import FrameRequest
from boxmot.structures import Detections, Frame, Tracks
from boxmot.trackers import TrackerRequirements, TrackerSpec
from tests.unit.engine.service.test_service import _aabb_frame, _FakeFactory, _settings


class _TimedTracker(ByteTrack):
    """Record the intervals resolved by the production tracker boundary."""

    requirements = TrackerRequirements(frame=True)

    def __init__(self, *, variable_dt: bool = False) -> None:
        super().__init__(asso_func="iou", variable_dt=variable_dt)
        self.intervals: list[float | None] = []
        self.calls: list[tuple[Detections, Frame | None]] = []

    def update(self, detections: Detections, frame: Frame | None = None, *, timestamp_s: float | None = None) -> Tracks:
        tracks = super().update(detections, frame, timestamp_s=timestamp_s)
        self.intervals.append(self._prediction_dt)
        self.calls.append((detections, frame))
        return tracks


def test_capture_timestamps_drive_intervals_and_retries_do_not_advance_time() -> None:
    tracker = _TimedTracker(variable_dt=True)

    def factory(spec: TrackerSpec) -> _TimedTracker:
        assert spec.option_dict["variable_dt"] is True
        return tracker

    path = "/v1/streams/camera/sessions/timed/frames"
    first = _aabb_frame(timestamp_s=10.0)
    second = _aabb_frame(frame_id=1, timestamp_s=10.04, detections=[])
    with TestClient(create_app(_settings(variable_dt=True), tracker_factory=factory)) as client:
        assert client.post(path, json=first).status_code == 200
        assert client.post(path, json=second).status_code == 200
        replay = client.post(path, json=second)
        assert replay.status_code == 200
        assert replay.json()["replayed"] is True
        assert client.post(path, json={**second, "timestamp_s": 10.05}).status_code == 409
        for timestamp in (10.04, 10.0):
            invalid = client.post(path, json=_aabb_frame(frame_id=2, timestamp_s=timestamp))
            assert invalid.status_code == 409
            assert "timestamp_s" in invalid.json()["detail"]
        assert client.post(path, json=_aabb_frame(frame_id=2, timestamp_s=10.14)).status_code == 200

    assert tracker.intervals[0] is None
    assert tracker.intervals[1:] == pytest.approx([0.04, 0.10])
    assert [frame.timestamp_s for _, frame in tracker.calls] == [10.0, 10.04, 10.14]


def test_variable_time_requires_first_timestamp_before_allocating_tracker() -> None:
    factory = _FakeFactory()
    with TestClient(create_app(_settings(variable_dt=True), tracker_factory=factory)) as client:
        response = client.post("/v1/streams/camera/sessions/timed/frames", json=_aabb_frame())

    assert response.status_code == 422
    assert "timestamp_s" in response.json()["detail"]
    assert factory.instances == []


def test_variable_time_rejects_missing_timestamp_without_advancing_session() -> None:
    tracker = _TimedTracker(variable_dt=True)
    with TestClient(create_app(_settings(variable_dt=True), tracker_factory=lambda spec: tracker)) as client:
        path = "/v1/streams/camera/sessions/timed/frames"
        assert client.post(path, json=_aabb_frame(timestamp_s=1.0)).status_code == 200
        response = client.post(path, json=_aabb_frame(frame_id=1))
        assert client.post(path, json=_aabb_frame(frame_id=1, timestamp_s=1.1)).status_code == 200

    assert response.status_code == 409
    assert "timestamp_s" in response.json()["detail"]
    assert tracker.intervals[0] is None
    assert tracker.intervals[1:] == pytest.approx([0.1])


def test_default_motion_is_unchanged_by_optional_timestamp_metadata() -> None:
    tracker = _TimedTracker()
    baseline = _TimedTracker()
    instances = iter((tracker, baseline))

    def factory(spec: TrackerSpec) -> _TimedTracker:
        assert spec.option_dict["variable_dt"] is False
        return next(instances)

    timestamps = [None, 10.0, 5.0, None, 4.0]
    with TestClient(create_app(_settings(), tracker_factory=factory)) as client:
        for frame_id, timestamp in enumerate(timestamps):
            frame = _aabb_frame(
                frame_id=frame_id,
                detections=[[10 + frame_id, 20, 30 + frame_id, 50, 0.9, 0]],
            )
            actual = client.post(
                "/v1/streams/camera/sessions/metadata/frames",
                json={**frame, "timestamp_s": timestamp},
            )
            expected = client.post("/v1/streams/camera/sessions/baseline/frames", json=frame)
            assert actual.status_code == expected.status_code == 200
            assert actual.json()["tracks"] == expected.json()["tracks"]

        retry = client.post(
            "/v1/streams/camera/sessions/metadata/frames",
            json={**frame, "timestamp_s": 4.0},
        )
        assert retry.status_code == 200
        assert retry.json()["replayed"] is True
        assert (
            client.post(
                "/v1/streams/camera/sessions/metadata/frames",
                json={**frame, "timestamp_s": 4.1},
            ).status_code
            == 409
        )

    assert tracker.intervals == baseline.intervals == [None] * len(timestamps)
    assert [frame.timestamp_s for _, frame in tracker.calls] == timestamps


@pytest.mark.parametrize("timestamp", [float("nan"), float("inf"), -float("inf")])
def test_capture_timestamps_must_be_finite(timestamp: float) -> None:
    with pytest.raises(ValidationError, match="finite"):
        FrameRequest(**_aabb_frame(timestamp_s=timestamp))


def test_non_kalman_service_tracker_accepts_capture_timestamps_as_metadata() -> None:
    factory = _FakeFactory()
    with TestClient(create_app(_settings(tracker_type="sfsort"), tracker_factory=factory)) as client:
        response = client.post(
            "/v1/streams/camera/sessions/timed/frames",
            json=_aabb_frame(timestamp_s=0.0),
        )
    assert response.status_code == 200
    assert factory.instances[0].calls[0][1].timestamp_s == 0.0
