from __future__ import annotations

from types import SimpleNamespace

import pytest

from boxmot.detectors import DetectorCapabilities
from boxmot.engine.tracking import workflow
from boxmot.engine.tracking.sinks import NullSink
from boxmot.reid import EncoderRequirements, ReIDEncoderSpec
from boxmot.structures import GeometryKind
from boxmot.trackers import TrackerCapabilities, TrackerFamily, TrackerRequirements


class _EmptySource:
    def __init__(self) -> None:
        self.closed = False

    def __iter__(self):
        return iter(())

    def close(self) -> None:
        self.closed = True


class _Detector:
    capabilities = DetectorCapabilities()

    def predict(self, frames):  # pragma: no cover - the source is deliberately empty
        raise AssertionError(f"Unexpected detector call for {len(frames)} frame(s).")


class _EmbeddingDetector(_Detector):
    capabilities = DetectorCapabilities(provides_embeddings=True)


class _Tracker:
    name = "botsort"
    supports_obb = True
    capabilities = TrackerCapabilities(
        family=TrackerFamily.BOX,
        geometry_kinds=frozenset({GeometryKind.AABB, GeometryKind.OBB}),
        accepts_embeddings=True,
        accepts_frame=True,
    )

    def __init__(self, *, generates_embeddings: bool) -> None:
        self.generates_embeddings = generates_embeddings
        self.requirements = TrackerRequirements(embeddings=True)
        self.configured_reid_specs: list[ReIDEncoderSpec] = []

    def configure_reid(self, spec: ReIDEncoderSpec) -> None:
        self.configured_reid_specs.append(spec)

    def update(self, detections, frame=None):  # pragma: no cover - the source is deliberately empty
        raise AssertionError("Unexpected tracker call.")

    def reset(self) -> None:
        return None


class _Encoder:
    embedding_dim = 4
    requirements = EncoderRequirements()

    def encode(self, frames, detections):  # pragma: no cover - the source is deliberately empty
        raise AssertionError("Unexpected encoder call.")


_PYTHON_REID_TRACKERS = (
    "boosttrack",
    "botsort",
    "deepocsort",
    "hybridsort",
    "occluboost",
    "strongsort",
)
_NATIVE_REID_TRACKERS = ("botsort", "occluboost")


def _args(*, tracker: str = "botsort", tracker_backend: str = "python") -> SimpleNamespace:
    return SimpleNamespace(
        geometry="aabb",
        tracker=tracker,
        tracker_backend=tracker_backend,
        reid="fixture-reid",
        device="cuda:2",
        half=True,
        per_class=False,
        classes=None,
        asso_func=None,
        segmentor=None,
    )


@pytest.mark.parametrize("tracker_name", _PYTHON_REID_TRACKERS)
def test_run_track_gives_python_reid_tracker_ownership_of_the_model(monkeypatch, tracker_name: str) -> None:
    source = _EmptySource()
    created_tracker_specs = []
    created_trackers: list[_Tracker] = []
    resolved_reid_spec = ReIDEncoderSpec(
        "onnx",
        artifact="models/fixture.onnx",
        artifact_sha256="a" * 64,
        device="cuda:2",
        precision="fp16",
        options=(("batch_size", 7), ("embedding_dim", 13), ("image_size", (192, 96))),
        preprocessing="letterbox",
    )

    monkeypatch.setattr(workflow, "_reid_spec", lambda _args: resolved_reid_spec)

    def create_tracker(spec):
        created_tracker_specs.append(spec)
        tracker = _Tracker(generates_embeddings=True)
        created_trackers.append(tracker)
        return tracker

    monkeypatch.setattr(workflow, "create_tracker", create_tracker)

    def reject_external_encoder(_spec):
        raise AssertionError("Python trackers must generate missing embeddings internally.")

    monkeypatch.setattr(workflow, "create_reid_encoder", reject_external_encoder)

    run = workflow.run_track(
        _args(tracker=tracker_name),
        detector=_Detector(),
        source=source,
        sinks=(NullSink(),),
    )

    assert run.summary.frames == 0
    assert source.closed
    assert len(created_tracker_specs) == 1
    assert created_tracker_specs[0].name == tracker_name
    assert created_tracker_specs[0].backend == "python"
    assert created_tracker_specs[0].option_dict == {}
    assert created_trackers[0].configured_reid_specs == [resolved_reid_spec]
    assert run.summary.startup_timings_ms["reid_load"] == 0.0


@pytest.mark.parametrize("tracker_name", _NATIVE_REID_TRACKERS)
def test_run_track_gives_native_reid_tracker_ownership_of_the_model(monkeypatch, tracker_name: str) -> None:
    source = _EmptySource()
    created_tracker_specs = []
    created_trackers: list[_Tracker] = []
    resolved_reid_spec = ReIDEncoderSpec(
        "onnx",
        artifact="models/fixture.onnx",
        device="cpu",
        precision="fp32",
    )

    monkeypatch.setattr(workflow, "_reid_spec", lambda _args: resolved_reid_spec)

    def create_tracker(spec):
        created_tracker_specs.append(spec)
        tracker = _Tracker(generates_embeddings=True)
        created_trackers.append(tracker)
        return tracker

    monkeypatch.setattr(workflow, "create_tracker", create_tracker)

    def reject_external_encoder(_spec):
        raise AssertionError("Native tracker adapters must generate missing embeddings internally.")

    monkeypatch.setattr(workflow, "create_reid_encoder", reject_external_encoder)

    run = workflow.run_track(
        _args(tracker=tracker_name, tracker_backend="cpp"),
        detector=_Detector(),
        source=source,
        sinks=(NullSink(),),
    )

    assert run.summary.frames == 0
    assert source.closed
    assert len(created_tracker_specs) == 1
    assert created_tracker_specs[0].name == tracker_name
    assert created_tracker_specs[0].backend == "cpp"
    assert created_tracker_specs[0].option_dict == {}
    assert created_trackers[0].configured_reid_specs == [resolved_reid_spec]
    assert run.summary.startup_timings_ms["reid_load"] == 0.0


def test_run_track_keeps_native_reid_owned_by_a_python_tracker(monkeypatch) -> None:
    source = _EmptySource()
    created_tracker_specs = []
    created_trackers: list[_Tracker] = []
    resolved_reid_spec = ReIDEncoderSpec(
        "native",
        artifact="models/fixture.onnx",
        device="cpu",
        precision="fp32",
    )

    monkeypatch.setattr(workflow, "_reid_spec", lambda _args: resolved_reid_spec)

    def create_tracker(spec):
        created_tracker_specs.append(spec)
        tracker = _Tracker(generates_embeddings=True)
        created_trackers.append(tracker)
        return tracker

    monkeypatch.setattr(workflow, "create_tracker", create_tracker)

    def reject_external_encoder(_spec):
        raise AssertionError("Python trackers must own native ReID fallback too.")

    monkeypatch.setattr(workflow, "create_reid_encoder", reject_external_encoder)

    workflow.run_track(
        _args(),
        detector=_Detector(),
        source=source,
        sinks=(NullSink(),),
    )

    assert len(created_tracker_specs) == 1
    assert created_tracker_specs[0].option_dict == {}
    assert created_trackers[0].configured_reid_specs == [resolved_reid_spec]


def test_run_track_keeps_reid_upstream_for_a_caller_owned_tracker(monkeypatch) -> None:
    source = _EmptySource()
    tracker = _Tracker(generates_embeddings=True)
    resolved_reid_spec = ReIDEncoderSpec(
        "onnx",
        artifact="models/fixture.onnx",
        device="cpu",
        precision="fp32",
    )
    created_encoder_specs: list[ReIDEncoderSpec] = []

    monkeypatch.setattr(workflow, "_reid_spec", lambda _args: resolved_reid_spec)
    monkeypatch.setattr(
        workflow,
        "create_reid_encoder",
        lambda spec: created_encoder_specs.append(spec) or _Encoder(),
    )

    workflow.run_track(
        _args(),
        detector=_Detector(),
        tracker=tracker,
        source=source,
        sinks=(NullSink(),),
    )

    assert tracker.configured_reid_specs == []
    assert created_encoder_specs == [resolved_reid_spec]


@pytest.mark.parametrize(
    ("tracker_backend", "generates_embeddings"),
    (("python", True), ("cpp", True)),
)
def test_run_track_does_not_resolve_reid_when_detector_provides_embeddings(
    monkeypatch,
    tracker_backend: str,
    generates_embeddings: bool,
) -> None:
    source = _EmptySource()
    created_tracker_specs = []
    created_trackers: list[_Tracker] = []

    def reject_reid_resolution(_args):
        raise AssertionError("Precomputed detector embeddings must bypass ReID resolution.")

    monkeypatch.setattr(workflow, "_reid_spec", reject_reid_resolution)

    def create_tracker(spec):
        created_tracker_specs.append(spec)
        tracker = _Tracker(generates_embeddings=generates_embeddings)
        created_trackers.append(tracker)
        return tracker

    monkeypatch.setattr(workflow, "create_tracker", create_tracker)

    workflow.run_track(
        _args(tracker_backend=tracker_backend),
        detector=_EmbeddingDetector(),
        source=source,
        sinks=(NullSink(),),
    )

    assert len(created_tracker_specs) == 1
    assert created_tracker_specs[0].option_dict == {}
    assert created_trackers[0].configured_reid_specs == []
