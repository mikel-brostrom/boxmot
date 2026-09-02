from __future__ import annotations

import asyncio
import base64
import subprocess
import sys
import textwrap
import threading

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import boxmot.engine.service.manager as service_manager
from boxmot.engine.service.app import create_app
from boxmot.engine.service.config import (
    CPU_SERVICE_TRACKERS,
    REID_SERVICE_TRACKERS,
    SERVICE_TRACKERS_BY_PROFILE,
    ServiceSettings,
)
from boxmot.engine.service.manager import FrameConflictError, TrackerManager
from boxmot.engine.service.models import FrameRequest


class _FakeTracker:
    uses_img = True
    uses_embs = False
    supports_masks = False

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


class _DetectionsOnlyTracker:
    uses_img = False
    uses_embs = False
    supports_masks = False

    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []
        self.reset_calls = 0

    def update(self, detections: np.ndarray) -> np.ndarray:
        self.calls.append(detections.copy())
        return np.empty((0, detections.shape[1] + 2), dtype=np.float32)

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


def _encoded_image(
    *,
    width: int = 32,
    height: int = 24,
    value: int = 64,
    extension: str = ".png",
) -> tuple[str, np.ndarray]:
    image = np.full((height, width, 3), value, dtype=np.uint8)
    encoded_ok, encoded = cv2.imencode(extension, image)
    assert encoded_ok
    return base64.b64encode(encoded.tobytes()).decode("ascii"), image


def test_health_and_readiness_report_tracker_capacity() -> None:
    factory = _FakeFactory()
    with TestClient(create_app(_settings(max_streams=7), tracker_factory=factory)) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        assert client.get("/readyz").json() == {
            "status": "ready",
            "profile": "cpu",
            "tracker": "bytetrack",
            "device": "cpu",
            "requires_image": False,
            "active_streams": 0,
            "max_streams": 7,
        }


def test_readiness_reports_gpu_image_requirement_and_device() -> None:
    settings = _settings(profile="gpu", tracker_type="botsort", device="cuda:0")

    with TestClient(create_app(settings, tracker_factory=_FakeFactory())) as client:
        assert client.get("/readyz").json() == {
            "status": "ready",
            "profile": "gpu",
            "tracker": "botsort",
            "device": "cuda:0",
            "requires_image": True,
            "active_streams": 0,
            "max_streams": 4,
        }


def test_service_profiles_have_disjoint_expected_tracker_sets() -> None:
    assert CPU_SERVICE_TRACKERS == ("bytetrack", "ocsort", "sfsort")
    assert REID_SERVICE_TRACKERS == (
        "strongsort",
        "botsort",
        "deepocsort",
        "hybridsort",
        "boosttrack",
        "occluboost",
    )
    assert SERVICE_TRACKERS_BY_PROFILE == {
        "cpu": CPU_SERVICE_TRACKERS,
        "gpu": REID_SERVICE_TRACKERS,
    }
    assert set(CPU_SERVICE_TRACKERS).isdisjoint(REID_SERVICE_TRACKERS)


@pytest.mark.parametrize(
    ("overrides", "detail"),
    [
        ({"profile": "other"}, "Unsupported service profile"),
        ({"profile": "cpu", "tracker_type": "botsort"}, "not available in the 'cpu'"),
        ({"profile": "gpu", "tracker_type": "bytetrack"}, "not available in the 'gpu'"),
        ({"device": " "}, "device must not be empty"),
        ({"reid_weights": " "}, "weights must not be empty"),
    ],
)
def test_service_settings_reject_invalid_profile_configuration(overrides, detail) -> None:
    with pytest.raises(ValueError, match=detail):
        _settings(**overrides)


def test_gpu_environment_defaults_and_overrides(monkeypatch) -> None:
    environment_names = (
        "BOXMOT_SERVICE_PROFILE",
        "BOXMOT_SERVICE_TRACKER",
        "BOXMOT_SERVICE_DEVICE",
        "BOXMOT_SERVICE_HALF",
        "BOXMOT_SERVICE_REID_WEIGHTS",
        "BOXMOT_SERVICE_MAX_CONCURRENT_UPDATES",
    )
    for name in environment_names:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BOXMOT_SERVICE_PROFILE", " GPU ")

    defaults = ServiceSettings.from_env()

    assert defaults.profile == "gpu"
    assert defaults.tracker_type == "botsort"
    assert defaults.device == "0"
    assert defaults.half is True
    assert defaults.max_concurrent_updates == 1
    assert defaults.requires_image is True

    monkeypatch.setenv("BOXMOT_SERVICE_TRACKER", "boosttrack")
    monkeypatch.setenv("BOXMOT_SERVICE_DEVICE", "cuda:1")
    monkeypatch.setenv("BOXMOT_SERVICE_HALF", "off")
    monkeypatch.setenv("BOXMOT_SERVICE_REID_WEIGHTS", "/models/reid.pt")
    monkeypatch.setenv("BOXMOT_SERVICE_MAX_CONCURRENT_UPDATES", "3")

    overridden = ServiceSettings.from_env()

    assert overridden.tracker_type == "boosttrack"
    assert overridden.device == "cuda:1"
    assert overridden.half is False
    assert overridden.reid_weights == "/models/reid.pt"
    assert overridden.max_concurrent_updates == 3


def test_environment_rejects_invalid_boolean(monkeypatch) -> None:
    monkeypatch.setenv("BOXMOT_SERVICE_HALF", "sometimes")

    with pytest.raises(ValueError, match="BOXMOT_SERVICE_HALF must be a boolean"):
        ServiceSettings.from_env()


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


def test_cpu_motion_only_tracker_receives_detections_without_a_dummy_image() -> None:
    tracker = _DetectionsOnlyTracker()
    manager = TrackerManager(_settings(), tracker_factory=lambda _: tracker)
    key = ("camera", "run")

    async def scenario() -> None:
        result = await manager.process(key, FrameRequest(**_aabb_frame()))
        state = manager._states[key]

        assert result.tracks == ()
        assert state.input_adapter.uses_img is False
        assert state.image is None
        await manager.close()

    asyncio.run(scenario())

    assert len(tracker.calls) == 1
    np.testing.assert_array_equal(
        tracker.calls[0],
        np.array([[10, 20, 30, 50, 0.9, 0]], dtype=np.float32),
    )


def test_cpu_profile_uses_a_supplied_real_image_when_present() -> None:
    factory = _FakeFactory()
    encoded, source = _encoded_image()
    frame = _aabb_frame(width=32, height=24, image_base64=encoded)

    with TestClient(create_app(_settings(), tracker_factory=factory)) as client:
        response = client.post("/v1/streams/a/sessions/b/frames", json=frame)

    assert response.status_code == 200
    assert factory.instances[0].uses_img is True
    decoded = factory.instances[0].calls[0][1]
    assert decoded.flags.c_contiguous
    np.testing.assert_array_equal(decoded, source)


def test_gpu_profile_requires_an_image_on_every_frame_even_without_detections() -> None:
    factory = _FakeFactory()
    settings = _settings(profile="gpu", tracker_type="botsort", device="cuda:0")
    encoded, _ = _encoded_image()
    first = _aabb_frame(width=32, height=24, image_base64=encoded)
    second = _aabb_frame(
        frame_id=1,
        width=32,
        height=24,
        detections=[],
        image_base64=encoded,
    )

    with TestClient(create_app(settings, tracker_factory=factory)) as client:
        missing_first = client.post(
            "/v1/streams/a/sessions/b/frames",
            json=_aabb_frame(width=32, height=24, detections=[]),
        )
        accepted_first = client.post("/v1/streams/a/sessions/b/frames", json=first)
        missing_second = client.post(
            "/v1/streams/a/sessions/b/frames",
            json=_aabb_frame(frame_id=1, width=32, height=24, detections=[]),
        )
        accepted_second = client.post("/v1/streams/a/sessions/b/frames", json=second)

    assert missing_first.status_code == 422
    assert "including on frames with no detections" in missing_first.json()["detail"]
    assert accepted_first.status_code == 200
    assert missing_second.status_code == 422
    assert accepted_second.status_code == 200
    assert len(factory.instances) == 1
    assert len(factory.instances[0].calls) == 2


@pytest.mark.parametrize("extension", [".jpg", ".png"])
def test_gpu_profile_decodes_supported_images_with_exact_dimensions(extension) -> None:
    factory = _FakeFactory()
    settings = _settings(profile="gpu", tracker_type="botsort", device="cuda:0")
    encoded, source = _encoded_image(extension=extension)
    frame = _aabb_frame(width=32, height=24, image_base64=encoded)

    with TestClient(create_app(settings, tracker_factory=factory)) as client:
        response = client.post("/v1/streams/a/sessions/b/frames", json=frame)

    assert response.status_code == 200
    decoded = factory.instances[0].calls[0][1]
    assert decoded.shape == (24, 32, 3)
    assert decoded.dtype == np.uint8
    assert decoded.flags.c_contiguous
    if extension == ".png":
        np.testing.assert_array_equal(decoded, source)


@pytest.mark.parametrize(
    ("image_base64", "detail"),
    [
        ("not-base64!", "not valid base64"),
        (base64.b64encode(b"not an image").decode("ascii"), "JPEG or PNG"),
        ("", "non-empty JPEG or PNG"),
        ("not-ascii-☃", "ASCII base64"),
    ],
)
def test_invalid_encoded_images_return_422_without_creating_a_tracker(image_base64, detail) -> None:
    factory = _FakeFactory()
    settings = _settings(profile="gpu", tracker_type="botsort", device="cuda:0")

    with TestClient(create_app(settings, tracker_factory=factory)) as client:
        response = client.post(
            "/v1/streams/a/sessions/b/frames",
            json=_aabb_frame(width=32, height=24, detections=[], image_base64=image_base64),
        )

    assert response.status_code == 422
    assert detail in response.json()["detail"]
    assert factory.instances == []


def test_tracker_creation_failure_returns_managed_500_without_retaining_state() -> None:
    def failing_factory(frame_rate: int):
        raise RuntimeError(f"cannot create tracker at {frame_rate} FPS")

    application = create_app(_settings(), tracker_factory=failing_factory)
    with TestClient(application, raise_server_exceptions=False) as client:
        response = client.post(
            "/v1/streams/a/sessions/b/frames",
            json=_aabb_frame(),
        )

    assert response.status_code == 500
    assert response.json()["detail"] == ("Tracker creation failed; verify the selected tracker and service profile.")
    assert application.state.tracker_manager._states == {}


def test_decoded_image_dimensions_must_exactly_match_request_metadata() -> None:
    factory = _FakeFactory()
    settings = _settings(profile="gpu", tracker_type="botsort", device="cuda:0")
    encoded, _ = _encoded_image(width=32, height=24)

    with TestClient(create_app(settings, tracker_factory=factory)) as client:
        response = client.post(
            "/v1/streams/a/sessions/b/frames",
            json=_aabb_frame(width=31, height=24, detections=[], image_base64=encoded),
        )

    assert response.status_code == 422
    assert "expected (31, 24), got (32, 24)" in response.json()["detail"]
    assert factory.instances == []


def test_gpu_retry_identity_includes_encoded_image_bytes() -> None:
    factory = _FakeFactory()
    settings = _settings(profile="gpu", tracker_type="botsort", device="cuda:0")
    first_image, _ = _encoded_image(value=32)
    different_image, _ = _encoded_image(value=224)
    first_frame = _aabb_frame(width=32, height=24, image_base64=first_image)

    with TestClient(create_app(settings, tracker_factory=factory)) as client:
        first = client.post("/v1/streams/a/sessions/b/frames", json=first_frame)
        retry = client.post("/v1/streams/a/sessions/b/frames", json=first_frame)
        conflict = client.post(
            "/v1/streams/a/sessions/b/frames",
            json=_aabb_frame(width=32, height=24, image_base64=different_image),
        )
        next_frame = client.post(
            "/v1/streams/a/sessions/b/frames",
            json=_aabb_frame(
                frame_id=1,
                width=32,
                height=24,
                image_base64=different_image,
            ),
        )

    assert first.status_code == 200
    assert retry.status_code == 200
    assert retry.json()["replayed"] is True
    assert conflict.status_code == 409
    assert "different input" in conflict.json()["detail"]
    assert next_frame.status_code == 200
    assert len(factory.instances[0].calls) == 2


def test_gpu_manager_shares_one_prebuilt_reid_backend_across_trackers(monkeypatch) -> None:
    settings = _settings(profile="gpu", tracker_type="botsort", device="cuda:0")
    shared_model = object()
    model_factory_calls = []
    tracker_calls = []

    def model_factory(received_settings):
        model_factory_calls.append(received_settings)
        return shared_model

    def fake_create_tracker(tracker_type, **kwargs):
        tracker_calls.append((tracker_type, kwargs))
        return _FakeTracker(len(tracker_calls), kwargs["tracker_kwargs"]["frame_rate"])

    monkeypatch.setattr(service_manager, "create_tracker", fake_create_tracker)
    manager = TrackerManager(settings, reid_model_factory=model_factory)

    first = manager._tracker_factory(24)
    second = manager._tracker_factory(30)

    assert model_factory_calls == [settings]
    assert first is not second
    assert [call[0] for call in tracker_calls] == ["botsort", "botsort"]
    assert [call[1]["tracker_kwargs"] for call in tracker_calls] == [
        {"frame_rate": 24},
        {"frame_rate": 30},
    ]
    assert all(call[1]["reid_model"] is shared_model for call in tracker_calls)
    assert all(call[1]["warmup_model"] is False for call in tracker_calls)


def test_custom_gpu_tracker_factory_skips_shared_reid_backend_construction() -> None:
    factory = _FakeFactory()

    def unexpected_model_factory(settings):
        pytest.fail(f"ReID model factory unexpectedly called for {settings}")

    manager = TrackerManager(
        _settings(profile="gpu", tracker_type="botsort", device="cuda:0"),
        tracker_factory=factory,
        reid_model_factory=unexpected_model_factory,
    )

    assert manager._tracker_factory(25) is factory.instances[0]


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
        second = asyncio.create_task(manager.process(("camera", "run"), FrameRequest(**_aabb_frame(frame_id=1))))
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
        second = asyncio.create_task(manager.process(("camera", "run"), FrameRequest(**_aabb_frame(frame_id=1))))
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
        ("sfsort", "aabb", [[10, 20, 60, 120, 0.95, 0]], 8),
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


def test_supported_service_trackers_do_not_import_torch() -> None:
    """Keep the detection-only service usable in its Torch-free image."""

    script = textwrap.dedent(
        """
        import builtins

        original_import = builtins.__import__

        def import_without_torch(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise AssertionError(f"service tracker imported {name}")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = import_without_torch

        import numpy as np

        from boxmot.trackers.registry import create_tracker

        image = np.zeros((120, 160, 3), dtype=np.uint8)
        cases = (
            ("bytetrack", np.array([[10, 20, 60, 100, 0.95, 0]], dtype=np.float32)),
            ("ocsort", np.array([[35, 60, 50, 80, 0.1, 0.95, 0]], dtype=np.float32)),
            ("sfsort", np.array([[10, 20, 60, 100, 0.95, 0]], dtype=np.float32)),
        )
        for tracker_type, detections in cases:
            tracker_kwargs = {"frame_rate": 30} if tracker_type == "bytetrack" else None
            tracker = create_tracker(
                tracker_type,
                per_class=True,
                tracker_backend="python",
                tracker_kwargs=tracker_kwargs,
            )
            tracker.update(detections, image)
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
