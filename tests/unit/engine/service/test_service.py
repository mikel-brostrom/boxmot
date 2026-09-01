from __future__ import annotations

import asyncio
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient

from boxmot.engine.service.app import create_app
from boxmot.engine.service.config import ServiceSettings
from boxmot.engine.service.manager import FrameConflictError, TrackerManager
from boxmot.engine.service.models import FrameRequest


class _FakeTracker:
    def __init__(self, instance_id: int, frame_rate: int) -> None:
        self.instance_id = instance_id
        self.frame_rate = frame_rate
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []
        self.reset_calls = 0

    def update(self, detections: np.ndarray, image: np.ndarray) -> np.ndarray:
        self.calls.append((detections.copy(), image))
        geometry_cols = detections.shape[1] - 2
        output = np.empty((len(detections), detections.shape[1] + 2), dtype=np.float32)
        if len(detections):
            output[:, :geometry_cols] = detections[:, :geometry_cols]
            output[:, geometry_cols] = self.instance_id
            output[:, geometry_cols + 1] = detections[:, geometry_cols]
            output[:, geometry_cols + 2] = detections[:, geometry_cols + 1]
            output[:, geometry_cols + 3] = np.arange(len(detections))
        return output

    def reset(self) -> None:
        self.reset_calls += 1


class _FakeFactory:
    def __init__(self) -> None:
        self.instances: list[_FakeTracker] = []

    def __call__(self, frame_rate: int) -> _FakeTracker:
        tracker = _FakeTracker(len(self.instances) + 1, frame_rate)
        self.instances.append(tracker)
        return tracker


def _settings(**overrides) -> ServiceSettings:
    values = {
        "tracker_type": "bytetrack",
        "max_streams": 4,
        "stream_ttl_seconds": 60.0,
        "max_detections_per_frame": 10,
    }
    values.update(overrides)
    return ServiceSettings(**values)


def _aabb_frame(frame_id: int = 0, detections=None, **overrides) -> dict:
    frame = {
        "frame_id": frame_id,
        "width": 640,
        "height": 480,
        "frame_rate": 25,
        "box_type": "aabb",
        "detections": detections if detections is not None else [[10, 20, 30, 50, 0.9, 0]],
    }
    frame.update(overrides)
    return frame


def test_health_and_readiness_report_tracker_capacity() -> None:
    factory = _FakeFactory()
    with TestClient(create_app(_settings(max_streams=7), tracker_factory=factory)) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        assert client.get("/readyz").json() == {
            "status": "ready",
            "tracker": "bytetrack",
            "active_streams": 0,
            "max_streams": 7,
        }


def test_service_tracks_aabb_detections_with_isolated_stream_state() -> None:
    factory = _FakeFactory()
    with TestClient(create_app(_settings(), tracker_factory=factory)) as client:
        first = client.post(
            "/v1/streams/camera-1/sessions/run-1/frames",
            json=_aabb_frame(detections=[[10, 20, 30, 50, 0.9, 0], [40, 30, 60, 70, 0.8, 2]]),
        )
        second = client.post(
            "/v1/streams/camera-2/sessions/run-1/frames",
            json=_aabb_frame(),
        )

    assert first.status_code == 200
    assert first.json() == {
        "frame_id": 0,
        "next_frame_id": 1,
        "box_type": "aabb",
        "track_columns": [
            "x1",
            "y1",
            "x2",
            "y2",
            "id",
            "confidence",
            "class_id",
            "detection_index",
        ],
        "tracks": [
            [10.0, 20.0, 30.0, 50.0, 1, pytest.approx(0.9), 0, 0],
            [40.0, 30.0, 60.0, 70.0, 1, pytest.approx(0.8), 2, 1],
        ],
        "replayed": False,
    }
    assert second.status_code == 200
    assert len(factory.instances) == 2
    assert factory.instances[0].frame_rate == 25
    assert factory.instances[0].calls[0][1].shape == (480, 640, 3)
    assert factory.instances[0].calls[0][1].strides[:2] == (0, 0)


def test_empty_obb_frame_preserves_the_seven_column_detection_schema() -> None:
    factory = _FakeFactory()
    frame = _aabb_frame(box_type="obb", detections=[])

    with TestClient(create_app(_settings(), tracker_factory=factory)) as client:
        response = client.post("/v1/streams/a/sessions/b/frames", json=frame)

    assert response.status_code == 200
    assert response.json()["box_type"] == "obb"
    assert response.json()["tracks"] == []
    assert response.json()["track_columns"][:5] == ["cx", "cy", "w", "h", "angle"]
    assert factory.instances[0].calls[0][0].shape == (0, 7)


def test_exact_retry_is_replayed_but_conflicts_and_gaps_are_rejected() -> None:
    factory = _FakeFactory()
    application = create_app(_settings(), tracker_factory=factory)

    with TestClient(application) as client:
        first = client.post("/v1/streams/a/sessions/b/frames", json=_aabb_frame())
        retry = client.post("/v1/streams/a/sessions/b/frames", json=_aabb_frame())
        conflicting = client.post(
            "/v1/streams/a/sessions/b/frames",
            json=_aabb_frame(detections=[[11, 20, 30, 50, 0.9, 0]]),
        )
        gap = client.post("/v1/streams/a/sessions/b/frames", json=_aabb_frame(frame_id=2))
        next_frame = client.post(
            "/v1/streams/a/sessions/b/frames",
            json=_aabb_frame(frame_id=1, detections=[]),
        )

    assert first.status_code == 200
    assert retry.status_code == 200
    assert retry.json()["replayed"] is True
    assert conflicting.status_code == 409
    assert gap.status_code == 409
    assert next_frame.status_code == 200
    assert len(factory.instances[0].calls) == 2


def test_stream_contract_cannot_change_within_a_session() -> None:
    factory = _FakeFactory()
    with TestClient(create_app(_settings(), tracker_factory=factory)) as client:
        assert client.post("/v1/streams/a/sessions/b/frames", json=_aabb_frame()).status_code == 200
        response = client.post(
            "/v1/streams/a/sessions/b/frames",
            json=_aabb_frame(frame_id=1, width=1280),
        )

    assert response.status_code == 409
    assert "cannot change" in response.json()["detail"]
    assert len(factory.instances[0].calls) == 1


def test_new_or_expired_session_must_start_at_frame_zero() -> None:
    factory = _FakeFactory()
    with TestClient(create_app(_settings(), tracker_factory=factory)) as client:
        response = client.post(
            "/v1/streams/a/sessions/b/frames",
            json=_aabb_frame(frame_id=5),
        )

    assert response.status_code == 409
    assert "frame 0" in response.json()["detail"]
    assert factory.instances == []


@pytest.mark.parametrize(
    ("detections", "detail"),
    [
        ([[10, 20, 30, 50, 0.9]], "shape"),
        ([[10, 20, 10, 50, 0.9, 0]], "x2 > x1"),
        ([[10, 20, 30, 50, 1.1, 0]], "between 0 and 1"),
        ([[10, 20, 30, 50, 0.9, -1]], "non-negative integers"),
        ([[10, 20, 30, 50, 0.9, 0.5]], "non-negative integers"),
        ([[10**400, 20, 30, 50, 0.9, 0]], "Input should be a valid number"),
        ([[i, 20, i + 1, 50, 0.9, 0] for i in range(11)], "At most 10 detections"),
    ],
)
def test_invalid_detections_return_422_without_creating_a_tracker(detections, detail) -> None:
    factory = _FakeFactory()
    with TestClient(create_app(_settings(), tracker_factory=factory)) as client:
        response = client.post(
            "/v1/streams/a/sessions/b/frames",
            json=_aabb_frame(detections=detections),
        )

    assert response.status_code == 422
    assert detail in str(response.json()["detail"])
    assert factory.instances == []


def test_delete_releases_capacity_and_starts_fresh_state() -> None:
    factory = _FakeFactory()
    with TestClient(create_app(_settings(max_streams=1), tracker_factory=factory)) as client:
        assert client.post("/v1/streams/a/sessions/one/frames", json=_aabb_frame()).status_code == 200
        full = client.post("/v1/streams/b/sessions/two/frames", json=_aabb_frame())
        deleted = client.delete("/v1/streams/a/sessions/one")
        replacement = client.post("/v1/streams/b/sessions/two/frames", json=_aabb_frame())
        missing = client.delete("/v1/streams/a/sessions/one")

    assert full.status_code == 503
    assert full.headers["retry-after"] == "1"
    assert deleted.status_code == 204
    assert replacement.status_code == 200
    assert missing.status_code == 204
    assert len(factory.instances) == 2
    assert factory.instances[0].reset_calls == 1


def test_session_rejects_unbounded_cumulative_class_state() -> None:
    factory = _FakeFactory()
    settings = _settings(max_classes_per_stream=2)
    first = _aabb_frame(
        detections=[
            [10, 20, 30, 50, 0.9, 0],
            [40, 20, 60, 50, 0.8, 1],
        ]
    )

    with TestClient(create_app(settings, tracker_factory=factory)) as client:
        assert client.post("/v1/streams/a/sessions/b/frames", json=first).status_code == 200
        overflow = client.post(
            "/v1/streams/a/sessions/b/frames",
            json=_aabb_frame(frame_id=1, detections=[[10, 20, 30, 50, 0.9, 2]]),
        )
        accepted = client.post(
            "/v1/streams/a/sessions/b/frames",
            json=_aabb_frame(frame_id=1, detections=[[10, 20, 30, 50, 0.9, 1]]),
        )

    assert overflow.status_code == 422
    assert "at most 2 distinct class IDs" in overflow.json()["detail"]
    assert accepted.status_code == 200
    assert len(factory.instances[0].calls) == 2


def test_manager_serializes_updates_for_the_same_stream() -> None:
    entered = threading.Event()
    release = threading.Event()

    class _BlockingTracker(_FakeTracker):
        def __init__(self) -> None:
            super().__init__(1, 30)
            self.active_updates = 0
            self.max_active_updates = 0

        def update(self, detections: np.ndarray, image: np.ndarray) -> np.ndarray:
            self.active_updates += 1
            self.max_active_updates = max(self.max_active_updates, self.active_updates)
            if not self.calls:
                entered.set()
                assert release.wait(timeout=2)
            try:
                return super().update(detections, image)
            finally:
                self.active_updates -= 1

    tracker = _BlockingTracker()
    manager = TrackerManager(_settings(), tracker_factory=lambda _: tracker)

    async def scenario() -> None:
        first = asyncio.create_task(manager.process(("camera", "run"), FrameRequest(**_aabb_frame())))
        assert await asyncio.to_thread(entered.wait, 1)
        second = asyncio.create_task(
            manager.process(("camera", "run"), FrameRequest(**_aabb_frame(frame_id=1)))
        )
        await asyncio.sleep(0.02)
        assert len(tracker.calls) == 0
        release.set()
        await asyncio.gather(first, second)
        await manager.close()

    asyncio.run(scenario())

    assert len(tracker.calls) == 2
    assert tracker.max_active_updates == 1


def test_manager_bounds_concurrent_updates_across_streams() -> None:
    entered = threading.Event()
    release = threading.Event()
    guard = threading.Lock()
    entries = 0

    class _ProcessBlockingTracker(_FakeTracker):
        def update(self, detections: np.ndarray, image: np.ndarray) -> np.ndarray:
            nonlocal entries
            with guard:
                entries += 1
                entry_number = entries
            if entry_number == 1:
                entered.set()
                assert release.wait(timeout=2)
            return super().update(detections, image)

    instance_id = 0

    def factory(frame_rate: int) -> _ProcessBlockingTracker:
        nonlocal instance_id
        instance_id += 1
        return _ProcessBlockingTracker(instance_id, frame_rate)

    manager = TrackerManager(
        _settings(max_concurrent_updates=1),
        tracker_factory=factory,
    )

    async def scenario() -> None:
        first = asyncio.create_task(manager.process(("camera-1", "run"), FrameRequest(**_aabb_frame())))
        assert await asyncio.to_thread(entered.wait, 1)
        second = asyncio.create_task(manager.process(("camera-2", "run"), FrameRequest(**_aabb_frame())))
        await asyncio.sleep(0.02)
        with guard:
            assert entries == 1
        release.set()
        await asyncio.gather(first, second)
        await manager.close()

    asyncio.run(scenario())

    assert entries == 2


def test_cancellation_drains_tracker_thread_before_releasing_stream_lock() -> None:
    entered = threading.Event()
    release = threading.Event()

    class _CancellationTracker(_FakeTracker):
        def __init__(self) -> None:
            super().__init__(1, 30)
            self.entries = 0
            self.active_updates = 0
            self.max_active_updates = 0

        def update(self, detections: np.ndarray, image: np.ndarray) -> np.ndarray:
            self.entries += 1
            self.active_updates += 1
            self.max_active_updates = max(self.max_active_updates, self.active_updates)
            if self.entries == 1:
                entered.set()
                assert release.wait(timeout=2)
            try:
                return super().update(detections, image)
            finally:
                self.active_updates -= 1

    tracker = _CancellationTracker()
    manager = TrackerManager(_settings(), tracker_factory=lambda _: tracker)

    async def scenario() -> None:
        first = asyncio.create_task(manager.process(("camera", "run"), FrameRequest(**_aabb_frame())))
        assert await asyncio.to_thread(entered.wait, 1)
        first.cancel()
        second = asyncio.create_task(
            manager.process(("camera", "run"), FrameRequest(**_aabb_frame(frame_id=1)))
        )
        await asyncio.sleep(0.02)
        assert tracker.entries == 1
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        result = await second
        assert result.next_frame_id == 2
        await manager.close()

    asyncio.run(scenario())

    assert len(tracker.calls) == 2
    assert tracker.max_active_updates == 1


def test_expired_session_is_reset_and_cannot_continue_silently() -> None:
    now = [0.0]
    factory = _FakeFactory()
    manager = TrackerManager(
        _settings(stream_ttl_seconds=1.0),
        tracker_factory=factory,
        clock=lambda: now[0],
    )

    async def scenario() -> None:
        await manager.process(("camera", "run"), FrameRequest(**_aabb_frame()))
        now[0] = 2.0
        with pytest.raises(FrameConflictError, match="expired"):
            await manager.process(
                ("camera", "run"),
                FrameRequest(**_aabb_frame(frame_id=1)),
            )
        assert await manager.stats() == {"active_streams": 0, "max_streams": 4}
        await manager.close()

    asyncio.run(scenario())

    assert len(factory.instances) == 1
    assert factory.instances[0].reset_calls == 1


@pytest.mark.parametrize(
    ("tracker_type", "box_type", "detections", "column_count"),
    [
        ("bytetrack", "aabb", [[10, 20, 60, 120, 0.95, 0]], 8),
        ("ocsort", "obb", [[35, 70, 50, 100, 0.1, 0.95, 0]], 9),
    ],
)
def test_default_factory_processes_canonical_empty_and_nonempty_frames(
    tracker_type,
    box_type,
    detections,
    column_count,
) -> None:
    settings = _settings(tracker_type=tracker_type)
    first = _aabb_frame(box_type=box_type, detections=[])
    second = _aabb_frame(frame_id=1, box_type=box_type, detections=detections)

    with TestClient(create_app(settings)) as client:
        empty_response = client.post("/v1/streams/real/sessions/one/frames", json=first)
        tracked_response = client.post("/v1/streams/real/sessions/one/frames", json=second)

    assert empty_response.status_code == 200
    assert tracked_response.status_code == 200
    assert len(tracked_response.json()["track_columns"]) == column_count
