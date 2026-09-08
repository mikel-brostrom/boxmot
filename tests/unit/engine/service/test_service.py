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
import torch
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
from boxmot.reid import EncoderRequirements
from boxmot.structures import Detections, Frame, MaskBatch, Tracks
from boxmot.trackers import TrackerRequirements, TrackerSpec


class _FakeTracker:
    name = "FakeTracker"
    supports_obb = True
    requirements = TrackerRequirements(frame=True)

    def __init__(self, instance_id: int, frame_rate: int) -> None:
        self.instance_id = instance_id
        self.frame_rate = frame_rate
        self.calls: list[tuple[Detections, Frame | None]] = []
        self.reset_calls = 0

    def update(
        self,
        detections: Detections,
        frame: Frame | None = None,
    ) -> Tracks:
        self.calls.append((detections, frame))
        return Tracks(
            geometry=detections.geometry,
            track_ids=torch.arange(
                self.instance_id,
                self.instance_id + len(detections),
                dtype=torch.int64,
            ),
            scores=detections.scores,
            class_ids=detections.class_ids,
            detection_indices=torch.arange(len(detections), dtype=torch.int64),
            sample_id=detections.sample_id,
        )

    def reset(self) -> None:
        self.reset_calls += 1


class _DetectionsOnlyTracker:
    name = "DetectionsOnlyTracker"
    supports_obb = True
    requirements = TrackerRequirements()

    def __init__(self) -> None:
        self.calls: list[Detections] = []
        self.reset_calls = 0

    def update(self, detections: Detections, frame: Frame | None = None) -> Tracks:
        assert frame is None
        self.calls.append(detections)
        return Tracks(
            geometry=detections.geometry.select(torch.empty(0, dtype=torch.int64)),
            track_ids=torch.empty(0, dtype=torch.int64),
            scores=torch.empty(0, dtype=torch.float32),
            class_ids=torch.empty(0, dtype=torch.int64),
            detection_indices=torch.empty(0, dtype=torch.int64),
            sample_id=detections.sample_id,
        )

    def reset(self) -> None:
        self.reset_calls += 1


class _FakeFactory:
    def __init__(self) -> None:
        self.instances: list[_FakeTracker] = []

    def __call__(self, spec: TrackerSpec) -> _FakeTracker:
        tracker = _FakeTracker(len(self.instances) + 1, int(spec.option_dict.get("frame_rate", 30)))
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


def _image_frame(image_base64: str, **overrides) -> dict:
    """Build an image request whose dimensions come from the encoded frame."""

    frame = _aabb_frame(image_base64=image_base64)
    del frame["width"]
    del frame["height"]
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
        ({"asso_func": "overlap"}, "Unsupported association function"),
        ({"variable_dt": "true"}, "variable_dt must be a boolean"),
        ({"tracker_type": "sfsort", "variable_dt": True}, "SFSORT does not support"),
    ],
)
def test_service_settings_reject_invalid_profile_configuration(overrides, detail) -> None:
    with pytest.raises(ValueError, match=detail):
        _settings(**overrides)


def test_gpu_environment_defaults_and_overrides(monkeypatch) -> None:
    environment_names = (
        "BOXMOT_SERVICE_PROFILE",
        "BOXMOT_SERVICE_TRACKER",
        "BOXMOT_SERVICE_ASSO_FUNC",
        "BOXMOT_VARIABLE_DT",
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
    assert defaults.asso_func == "iou"
    assert defaults.variable_dt is False
    assert defaults.device == "0"
    assert defaults.half is True
    assert defaults.max_concurrent_updates == 1
    assert defaults.requires_image is True

    monkeypatch.setenv("BOXMOT_SERVICE_TRACKER", "boosttrack")
    monkeypatch.setenv("BOXMOT_SERVICE_ASSO_FUNC", "giou")
    monkeypatch.setenv("BOXMOT_VARIABLE_DT", "on")
    monkeypatch.setenv("BOXMOT_SERVICE_DEVICE", "cuda:1")
    monkeypatch.setenv("BOXMOT_SERVICE_HALF", "off")
    monkeypatch.setenv("BOXMOT_SERVICE_REID_WEIGHTS", "/models/reid.pt")
    monkeypatch.setenv("BOXMOT_SERVICE_MAX_CONCURRENT_UPDATES", "3")

    overridden = ServiceSettings.from_env()

    assert overridden.tracker_type == "boosttrack"
    assert overridden.asso_func == "giou"
    assert overridden.variable_dt is True
    assert overridden.device == "cuda:1"
    assert overridden.half is False
    assert overridden.reid_weights == "/models/reid.pt"
    assert overridden.max_concurrent_updates == 3


@pytest.mark.parametrize("name", ["BOXMOT_SERVICE_HALF", "BOXMOT_VARIABLE_DT"])
def test_environment_rejects_invalid_boolean(name, monkeypatch) -> None:
    monkeypatch.setenv(name, "sometimes")

    with pytest.raises(ValueError, match=f"{name} must be a boolean"):
        ServiceSettings.from_env()


def test_environment_rejects_noncanonical_tracker_name(monkeypatch) -> None:
    monkeypatch.setenv("BOXMOT_SERVICE_TRACKER", "ByteTrack")

    with pytest.raises(ValueError, match="Tracker 'ByteTrack' is not available"):
        ServiceSettings.from_env()


def test_environment_rejects_noncanonical_association_name(monkeypatch) -> None:
    monkeypatch.setenv("BOXMOT_SERVICE_ASSO_FUNC", "GIoU")

    with pytest.raises(ValueError, match="Unsupported association function 'GIoU'"):
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
            [40.0, 30.0, 60.0, 70.0, 2, pytest.approx(0.8), 2, 1],
        ],
        "replayed": False,
    }
    assert second.status_code == 200
    assert len(factory.instances) == 2
    assert factory.instances[0].frame_rate == 25
    received_frame = factory.instances[0].calls[0][1]
    assert isinstance(received_frame, Frame)
    assert received_frame.image.shape == (3, 480, 640)
    assert received_frame.image.is_contiguous()


def test_service_preserves_int64_class_ids_without_float32_rounding() -> None:
    factory = _FakeFactory()
    class_id = 2**40 + 123

    with TestClient(create_app(_settings(), tracker_factory=factory)) as client:
        response = client.post(
            "/v1/streams/camera/sessions/run/frames",
            json=_aabb_frame(detections=[[10, 20, 30, 50, 0.9, class_id]]),
        )

    assert response.status_code == 200
    assert factory.instances[0].calls[0][0].class_ids.tolist() == [class_id]
    assert response.json()["tracks"][0][6] == class_id


def test_cpu_motion_only_tracker_receives_detections_without_a_dummy_image() -> None:
    tracker = _DetectionsOnlyTracker()
    manager = TrackerManager(_settings(), tracker_factory=lambda _: tracker)
    key = ("camera", "run")

    async def scenario() -> None:
        result = await manager.process(key, FrameRequest(**_aabb_frame()))
        state = manager._states[key]

        assert result.tracks == ()
        assert state.pipeline._requirements.frame is False
        await manager.close()

    asyncio.run(scenario())

    assert len(tracker.calls) == 1
    np.testing.assert_allclose(
        tracker.calls[0].to_aabb_rows().numpy(),
        np.array([[10, 20, 30, 50, 0.9, 0]], dtype=np.float32),
    )


def test_cpu_profile_uses_a_supplied_real_image_when_present() -> None:
    factory = _FakeFactory()
    encoded, source = _encoded_image()
    frame = _image_frame(encoded)

    with TestClient(create_app(_settings(), tracker_factory=factory)) as client:
        response = client.post("/v1/streams/a/sessions/b/frames", json=frame)

    assert response.status_code == 200
    decoded = factory.instances[0].calls[0][1]
    assert isinstance(decoded, Frame)
    assert decoded.image.is_contiguous()
    np.testing.assert_array_equal(decoded.image.permute(1, 2, 0).numpy(), source[..., ::-1])


@pytest.mark.parametrize("omitted_dimensions", [("width",), ("height",), ("width", "height")])
def test_requests_without_images_require_both_dimensions(omitted_dimensions) -> None:
    factory = _FakeFactory()
    frame = _aabb_frame()
    for dimension in omitted_dimensions:
        del frame[dimension]

    with TestClient(create_app(_settings(), tracker_factory=factory)) as client:
        response = client.post("/v1/streams/a/sessions/b/frames", json=frame)

    assert response.status_code == 422
    assert factory.instances == []


def test_gpu_profile_requires_an_image_on_every_frame_even_without_detections() -> None:
    factory = _FakeFactory()
    settings = _settings(profile="gpu", tracker_type="botsort", device="cuda:0")
    encoded, _ = _encoded_image()
    first = _image_frame(encoded)
    second = _image_frame(encoded, frame_id=1, detections=[])

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
def test_gpu_profile_infers_dimensions_from_supported_images(extension) -> None:
    factory = _FakeFactory()
    settings = _settings(profile="gpu", tracker_type="botsort", device="cuda:0")
    encoded, source = _encoded_image(extension=extension)
    frame = _image_frame(encoded)

    with TestClient(create_app(settings, tracker_factory=factory)) as client:
        response = client.post("/v1/streams/a/sessions/b/frames", json=frame)

    assert response.status_code == 200
    decoded = factory.instances[0].calls[0][1]
    assert isinstance(decoded, Frame)
    assert decoded.image.shape == (3, 24, 32)
    assert decoded.image.dtype == torch.uint8
    assert decoded.image.is_contiguous()
    if extension == ".png":
        np.testing.assert_array_equal(decoded.image.permute(1, 2, 0).numpy(), source[..., ::-1])


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
            json=_image_frame(image_base64, detections=[]),
        )

    assert response.status_code == 422
    assert detail in response.json()["detail"]
    assert factory.instances == []


def test_tracker_creation_failure_returns_managed_500_without_retaining_state() -> None:
    def failing_factory(spec: TrackerSpec):
        raise RuntimeError(f"cannot create tracker from {spec}")

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


@pytest.mark.parametrize(
    ("declared_dimensions", "status_code"),
    [
        ({"width": 32}, 200),
        ({"height": 24}, 200),
        ({"width": 31}, 422),
        ({"height": 23}, 422),
    ],
)
def test_image_requests_validate_each_supplied_dimension(declared_dimensions, status_code) -> None:
    factory = _FakeFactory()
    encoded, _ = _encoded_image()

    with TestClient(create_app(_settings(), tracker_factory=factory)) as client:
        response = client.post(
            "/v1/streams/a/sessions/b/frames",
            json=_image_frame(encoded, **declared_dimensions),
        )

    assert response.status_code == status_code
    if status_code == 200:
        assert factory.instances[0].calls[0][1].image.shape == (3, 24, 32)
    else:
        assert "dimensions" in response.json()["detail"]
        assert factory.instances == []


@pytest.mark.parametrize("extension", [".jpg", ".png"])
def test_inferred_pixel_limit_is_checked_before_decoding(extension, monkeypatch) -> None:
    factory = _FakeFactory()
    encoded, _ = _encoded_image(extension=extension)

    def unexpected_decode(*args, **kwargs):
        pytest.fail("Images above the pixel limit must be rejected before decoding")

    monkeypatch.setattr(service_manager.cv2, "imdecode", unexpected_decode)
    with TestClient(create_app(_settings(max_frame_pixels=32 * 24 - 1), tracker_factory=factory)) as client:
        response = client.post(
            "/v1/streams/a/sessions/b/frames",
            json=_image_frame(encoded),
        )

    assert response.status_code == 422
    assert "pixel limit" in response.json()["detail"]
    assert factory.instances == []


@pytest.mark.parametrize(("width", "height"), [(32_769, 1), (1, 32_769)])
def test_inferred_dimensions_respect_axis_limits_before_decoding(width, height, monkeypatch) -> None:
    factory = _FakeFactory()
    encoded, _ = _encoded_image(width=width, height=height)

    def unexpected_decode(*args, **kwargs):
        pytest.fail("Images above an axis limit must be rejected before decoding")

    monkeypatch.setattr(service_manager.cv2, "imdecode", unexpected_decode)
    with TestClient(create_app(_settings(), tracker_factory=factory)) as client:
        response = client.post(
            "/v1/streams/a/sessions/b/frames",
            json=_image_frame(encoded),
        )

    assert response.status_code == 422
    assert "32768 pixels" in response.json()["detail"]
    assert factory.instances == []


def test_inferred_resolution_change_is_rejected_without_advancing_session() -> None:
    factory = _FakeFactory()
    settings = _settings(profile="gpu", tracker_type="botsort", device="cuda:0")
    encoded, _ = _encoded_image()
    larger_image, _ = _encoded_image(width=64, height=48)

    with TestClient(create_app(settings, tracker_factory=factory)) as client:
        first = client.post("/v1/streams/a/sessions/b/frames", json=_image_frame(encoded))
        changed = client.post(
            "/v1/streams/a/sessions/b/frames",
            json=_image_frame(larger_image, frame_id=1),
        )
        accepted = client.post(
            "/v1/streams/a/sessions/b/frames",
            json=_image_frame(encoded, frame_id=1),
        )

    assert first.status_code == 200
    assert changed.status_code == 409
    assert "cannot change" in changed.json()["detail"]
    assert accepted.status_code == 200
    assert accepted.json()["next_frame_id"] == 2
    assert len(factory.instances) == 1
    assert len(factory.instances[0].calls) == 2


def test_gpu_retry_identity_includes_encoded_image_bytes() -> None:
    factory = _FakeFactory()
    settings = _settings(profile="gpu", tracker_type="botsort", device="cuda:0")
    first_image, _ = _encoded_image(value=32)
    different_image, _ = _encoded_image(value=224)
    first_frame = _image_frame(first_image)

    with TestClient(create_app(settings, tracker_factory=factory)) as client:
        first = client.post("/v1/streams/a/sessions/b/frames", json=first_frame)
        retry = client.post("/v1/streams/a/sessions/b/frames", json=first_frame)
        explicit_retry = client.post(
            "/v1/streams/a/sessions/b/frames",
            json=_image_frame(first_image, width=32, height=24),
        )
        conflict = client.post(
            "/v1/streams/a/sessions/b/frames",
            json=_image_frame(different_image),
        )
        next_frame = client.post(
            "/v1/streams/a/sessions/b/frames",
            json=_image_frame(different_image, frame_id=1),
        )

    assert first.status_code == 200
    assert retry.status_code == 200
    assert retry.json()["replayed"] is True
    assert explicit_retry.status_code == 200
    assert explicit_retry.json()["replayed"] is True
    assert conflict.status_code == 409
    assert "different input" in conflict.json()["detail"]
    assert next_frame.status_code == 200
    assert len(factory.instances[0].calls) == 2


def test_gpu_manager_shares_one_prebuilt_encoder_across_decoupled_trackers(monkeypatch) -> None:
    settings = _settings(profile="gpu", tracker_type="botsort", device="cuda:0")
    encoder_factory_calls = []
    tracker_specs = []

    class _Encoder:
        embedding_dim = 4
        requirements = EncoderRequirements()

        def __init__(self) -> None:
            self.calls = []

        def encode(self, frames, detections):
            self.calls.append((frames, detections))
            return [torch.full((len(item), 4), 0.5, dtype=torch.float32) for item in detections]

    shared_encoder = _Encoder()

    def encoder_factory(received_settings):
        encoder_factory_calls.append(received_settings)
        return shared_encoder

    class _EmbeddingTracker(_FakeTracker):
        requirements = TrackerRequirements(embeddings=True, frame=True)

        def update(self, detections: Detections, frame: Frame | None = None) -> Tracks:
            assert detections.embeddings is not None
            return super().update(detections, frame)

    def fake_create_tracker(spec: TrackerSpec):
        tracker_specs.append(spec)
        return _EmbeddingTracker(len(tracker_specs), int(spec.option_dict["frame_rate"]))

    monkeypatch.setattr(service_manager, "create_tracker", fake_create_tracker)
    manager = TrackerManager(settings, encoder_factory=encoder_factory)
    encoded, _ = _encoded_image()

    async def scenario() -> None:
        await manager.process(
            ("one", "run"),
            FrameRequest(**_image_frame(encoded, frame_rate=24)),
        )
        await manager.process(
            ("two", "run"),
            FrameRequest(**_image_frame(encoded, frame_rate=30)),
        )
        await manager.close()

    asyncio.run(scenario())

    assert encoder_factory_calls == [settings]
    assert len(shared_encoder.calls) == 2
    assert [spec.name for spec in tracker_specs] == ["botsort", "botsort"]
    assert [spec.option_dict for spec in tracker_specs] == [
        {"asso_func": "iou", "frame_rate": 24, "variable_dt": False},
        {"asso_func": "iou", "frame_rate": 30, "variable_dt": False},
    ]


def test_shared_segmentor_runs_before_mask_aware_encoder_and_stream_tracker() -> None:
    events: list[str] = []

    class _Segmentor:
        def segment(self, frames, detections):
            events.append("segment")
            return [
                MaskBatch(torch.ones((len(items), frame.height, frame.width), dtype=torch.bool))
                for frame, items in zip(frames, detections)
            ]

    class _MaskAwareEncoder:
        embedding_dim = 4
        requirements = EncoderRequirements(masks=True)

        def encode(self, frames, detections):
            assert all(items.masks is not None for items in detections)
            events.append("encode")
            return [torch.ones((len(items), 4), dtype=torch.float32) for items in detections]

    class _EnrichedTracker(_FakeTracker):
        requirements = TrackerRequirements(embeddings=True, frame=True)

        def update(self, detections: Detections, frame: Frame | None = None) -> Tracks:
            assert detections.masks is not None
            assert detections.embeddings is not None
            events.append("track")
            return super().update(detections, frame)

    tracker = _EnrichedTracker(1, 25)
    manager = TrackerManager(
        _settings(),
        tracker_factory=lambda _spec: tracker,
        encoder=_MaskAwareEncoder(),
        segmentor=_Segmentor(),
    )

    async def scenario() -> None:
        await manager.process(("camera", "run"), FrameRequest(**_aabb_frame()))
        await manager.close()

    asyncio.run(scenario())

    assert events == ["segment", "encode", "track"]


def test_custom_gpu_tracker_factory_skips_shared_encoder_construction() -> None:
    factory = _FakeFactory()

    def unexpected_encoder_factory(settings):
        pytest.fail(f"encoder factory unexpectedly called for {settings}")

    manager = TrackerManager(
        _settings(profile="gpu", tracker_type="botsort", device="cuda:0"),
        tracker_factory=factory,
        encoder_factory=unexpected_encoder_factory,
    )

    spec = TrackerSpec(name="botsort", geometry="aabb", options=(("frame_rate", 25),))
    assert manager._tracker_factory(spec) is factory.instances[0]


def test_empty_obb_frame_preserves_the_seven_column_detection_schema() -> None:
    factory = _FakeFactory()
    frame = _aabb_frame(box_type="obb", detections=[])

    with TestClient(create_app(_settings(), tracker_factory=factory)) as client:
        response = client.post("/v1/streams/a/sessions/b/frames", json=frame)

    assert response.status_code == 200
    assert response.json()["box_type"] == "obb"
    assert response.json()["tracks"] == []
    assert response.json()["track_columns"][:5] == ["cx", "cy", "w", "h", "angle"]
    assert factory.instances[0].calls[0][0].to_obb_rows().shape == (0, 7)


@pytest.mark.parametrize("asso_func", ["iou", "giou", "diou", "ciou", "hmiou", "centroid"])
def test_all_association_functions_accept_obb_requests(asso_func) -> None:
    factory = _FakeFactory()
    frame = _aabb_frame(box_type="obb", detections=[])

    with TestClient(create_app(_settings(asso_func=asso_func), tracker_factory=factory)) as client:
        response = client.post("/v1/streams/a/sessions/b/frames", json=frame)

    assert response.status_code == 200
    assert response.json()["box_type"] == "obb"
    assert len(factory.instances) == 1


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

        def update(self, detections: Detections, frame: Frame | None = None) -> Tracks:
            self.active_updates += 1
            self.max_active_updates = max(self.max_active_updates, self.active_updates)
            if not self.calls:
                entered.set()
                assert release.wait(timeout=2)
            try:
                return super().update(detections, frame)
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
        def update(self, detections: Detections, frame: Frame | None = None) -> Tracks:
            nonlocal entries
            with guard:
                entries += 1
                entry_number = entries
            if entry_number == 1:
                entered.set()
                assert release.wait(timeout=2)
            return super().update(detections, frame)

    instance_id = 0

    def factory(spec: TrackerSpec) -> _ProcessBlockingTracker:
        nonlocal instance_id
        instance_id += 1
        return _ProcessBlockingTracker(instance_id, int(spec.option_dict.get("frame_rate", 30)))

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

        def update(self, detections: Detections, frame: Frame | None = None) -> Tracks:
            self.entries += 1
            self.active_updates += 1
            self.max_active_updates = max(self.max_active_updates, self.active_updates)
            if self.entries == 1:
                entered.set()
                assert release.wait(timeout=2)
            try:
                return super().update(detections, frame)
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


def test_supported_cpu_service_trackers_use_canonical_torch_structures() -> None:
    """The CPU service image includes Torch and exercises the v24 contract."""

    script = textwrap.dedent(
        """
        import torch

        from boxmot.structures import Boxes, Detections, Frame, OrientedBoxes, Tracks
        from boxmot.trackers import TrackerSpec, create_tracker

        frame = Frame(
            torch.zeros((3, 120, 160), dtype=torch.uint8),
            sample_id="sample-0",
            sequence_id="sequence-0",
            frame_index=0,
        )
        cases = (
            ("bytetrack", "aabb", Boxes(torch.tensor([[10, 20, 60, 100]], dtype=torch.float32))),
            ("ocsort", "obb", OrientedBoxes(torch.tensor([[35, 60, 50, 80, 0.1]], dtype=torch.float32))),
            ("sfsort", "aabb", Boxes(torch.tensor([[10, 20, 60, 100]], dtype=torch.float32))),
        )
        for tracker_type, geometry_mode, geometry in cases:
            options = (("frame_rate", 30),) if tracker_type == "bytetrack" else ()
            tracker = create_tracker(TrackerSpec(
                name=tracker_type,
                geometry=geometry_mode,
                per_class=True,
                options=options,
            ))
            detections = Detections(
                geometry=geometry,
                scores=torch.tensor([0.95], dtype=torch.float32),
                class_ids=torch.tensor([0], dtype=torch.int64),
                sample_id=frame.sample_id,
            )
            tracks = tracker.update(detections, frame if tracker.requirements.frame else None)
            assert isinstance(tracks, Tracks)
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
