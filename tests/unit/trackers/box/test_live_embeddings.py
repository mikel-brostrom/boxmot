"""Live appearance extraction shared by Python ReID-enabled trackers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from boxmot.components.timing import timing_event_sink
from boxmot.reid import EncoderRequirements, ReIDEncoderSpec
from boxmot.structures import Boxes, Detections, Frame, MaskBatch, OrientedBoxes
from boxmot.trackers.common.registry import TRACKER_DEFINITIONS, get_tracker_class

REID_TRACKER_NAMES = tuple(
    name for name, definition in TRACKER_DEFINITIONS.items() if definition.capabilities.accepts_embeddings
)
OPTIONAL_REID_TRACKER_NAMES = tuple(
    name for name in REID_TRACKER_NAMES if not TRACKER_DEFINITIONS[name].capabilities.requires_embeddings
)


class _ReIDModelSpy:
    """Small ReID double that records established backend inputs."""

    def __init__(self, features: np.ndarray) -> None:
        self.features = np.asarray(features, dtype=np.float32)
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []

    def get_features(self, geometry: np.ndarray, image: np.ndarray) -> np.ndarray:
        self.calls.append((geometry.copy(), image.copy()))
        return self.features[: len(geometry)].copy()


class _AppearanceEncoderSpy:
    """Canonical encoder double used by the full-spec configuration path."""

    embedding_dim = 3
    requirements = EncoderRequirements(masks=True)

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Frame, ...], tuple[Detections, ...]]] = []

    def encode(self, frames, detections) -> list[torch.Tensor]:
        frame_batch = tuple(frames)
        detection_batch = tuple(detections)
        self.calls.append((frame_batch, detection_batch))
        return [
            torch.eye(len(frame_detections), self.embedding_dim, dtype=torch.float32)
            for frame_detections in detection_batch
        ]


def _tracker_options(tracker_name: str) -> dict[str, object]:
    """Return cheap, deterministic settings while leaving ReID enabled."""
    common: dict[str, object] = {
        "det_thresh": 0.2,
        "min_hits": 1,
    }
    if tracker_name != "strongsort":
        common["use_embeddings"] = True
    common.update(
        {
            "boosttrack": {
                "use_cmc": False,
                "use_dlo_boost": False,
                "use_duo_boost": False,
            },
            "botsort": {
                "use_cmc": False,
                "track_high_thresh": 0.2,
                "new_track_thresh": 0.2,
            },
            "deepocsort": {"cmc_off": True},
            "hybridsort": {},
            "occluboost": {
                "use_cmc": False,
                "use_dlo_boost": False,
                "use_duo_boost": False,
                "gta_enabled": False,
                "instant_confirm_thresh": 0.2,
                "new_track_thresh": 0.2,
            },
            "strongsort": {"min_conf": 0.2, "n_init": 1},
        }[tracker_name]
    )
    return common


def _tracker(tracker_name: str, *, reid_model: _ReIDModelSpy | None, **kwargs: object):
    tracker_class = get_tracker_class(tracker_name)
    return tracker_class(
        **_tracker_options(tracker_name),
        reid_model=reid_model,
        **kwargs,
    )


def _detections(
    *,
    geometry: str = "aabb",
    embeddings: torch.Tensor | None = None,
    empty: bool = False,
) -> Detections:
    if geometry == "obb":
        values = (
            torch.empty((0, 5), dtype=torch.float32)
            if empty
            else torch.tensor(
                [[12, 13, 16, 14, 0.2], [32, 15, 16, 16, -0.3]],
                dtype=torch.float32,
            )
        )
        boxes = OrientedBoxes(values)
    else:
        values = (
            torch.empty((0, 4), dtype=torch.float32)
            if empty
            else torch.tensor(
                [[4, 5, 20, 21], [24, 7, 40, 23]],
                dtype=torch.float32,
            )
        )
        boxes = Boxes(values)
    count = len(boxes)
    return Detections(
        geometry=boxes,
        scores=torch.full((count,), 0.9, dtype=torch.float32),
        class_ids=torch.arange(count, dtype=torch.int64),
        sample_id="sequence/000001",
        embeddings=embeddings,
    )


def _frame() -> Frame:
    rgb = torch.empty((3, 32, 48), dtype=torch.uint8)
    rgb[0].fill_(11)
    rgb[1].fill_(22)
    rgb[2].fill_(33)
    return Frame(
        image=rgb,
        sample_id="sequence/000001",
        sequence_id="sequence",
        frame_index=1,
    )


@pytest.mark.parametrize("tracker_name", REID_TRACKER_NAMES)
@pytest.mark.parametrize("geometry", ("aabb", "obb"))
@pytest.mark.parametrize("frame_representation", ("canonical", "numpy"))
def test_reid_trackers_generate_missing_embeddings_from_geometry_and_bgr_frame(
    tracker_name: str,
    geometry: str,
    frame_representation: str,
) -> None:
    expected_features = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    model = _ReIDModelSpy(expected_features)
    tracker = _tracker(tracker_name, reid_model=model, is_obb=geometry == "obb")
    detections = _detections(geometry=geometry)
    frame = _frame() if frame_representation == "canonical" else np.full((32, 48, 3), (33, 22, 11), dtype=np.uint8)

    tracker.update(detections, frame)

    assert len(model.calls) == 1
    received_geometry, image = model.calls[0]
    np.testing.assert_array_equal(received_geometry, detections.geometry.values.numpy())
    np.testing.assert_array_equal(image[0, 0], np.array([33, 22, 11], dtype=np.uint8))
    assert image.shape == (32, 48, 3)
    assert image.flags.c_contiguous


def test_direct_live_reid_emits_component_timing_phases() -> None:
    model = _ReIDModelSpy(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32))
    tracker = _tracker("botsort", reid_model=model)
    events = []

    with timing_event_sink(events.append):
        tracker.update(_detections(), _frame())

    assert {(event.component, event.phase) for event in events} == {
        ("reid", "preprocess"),
        ("reid", "process"),
        ("reid", "postprocess"),
    }


def test_direct_live_reid_reuses_one_bgr_frame_for_inference_and_tracking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _ReIDModelSpy(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32))
    tracker = _tracker("botsort", reid_model=model)
    original = tracker._frame_to_bgr
    converted_images: list[np.ndarray] = []

    def record_conversion(frame: Frame) -> np.ndarray:
        converted = original(frame)
        converted_images.append(converted)
        return converted

    monkeypatch.setattr(tracker, "_frame_to_bgr", record_conversion)

    tracker.update(_detections(), _frame())

    assert len(converted_images) == 1


@pytest.mark.parametrize("tracker_name", REID_TRACKER_NAMES)
def test_reid_trackers_bypass_model_when_embeddings_are_precomputed(tracker_name: str) -> None:
    model = _ReIDModelSpy(np.array([[1, 0, 0]], dtype=np.float32))
    tracker = _tracker(tracker_name, reid_model=model)
    supplied = torch.tensor([[0, 0, 1], [1, 0, 0]], dtype=torch.float32)

    tracker.update(_detections(embeddings=supplied), _frame())

    assert model.calls == []


@pytest.mark.parametrize("tracker_name", REID_TRACKER_NAMES)
def test_reid_trackers_require_frame_only_when_live_embeddings_are_needed(tracker_name: str) -> None:
    model = _ReIDModelSpy(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32))
    tracker = _tracker(tracker_name, reid_model=model)

    with pytest.raises(ValueError, match="[Ff]rame"):
        tracker.update(_detections())

    assert model.calls == []


@pytest.mark.parametrize("tracker_name", REID_TRACKER_NAMES)
def test_reid_trackers_skip_live_model_for_empty_batches(tracker_name: str) -> None:
    model = _ReIDModelSpy(np.empty((0, 3), dtype=np.float32))
    tracker = _tracker(tracker_name, reid_model=model)

    tracks = tracker.update(_detections(empty=True), _frame())

    assert len(tracks) == 0
    assert model.calls == []


@pytest.mark.parametrize("tracker_name", REID_TRACKER_NAMES)
def test_reid_trackers_build_default_backend_lazily_with_shared_options(
    tracker_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import boxmot.reid.core as reid_core

    model = _ReIDModelSpy(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32))
    constructor_calls: list[dict[str, object]] = []

    class _ReIDRuntime:
        def __init__(self, **kwargs: object) -> None:
            constructor_calls.append(kwargs)
            self.model = model

    monkeypatch.setattr(reid_core, "ReID", _ReIDRuntime)
    weights = Path("custom-reid.pt")
    tracker = _tracker(
        tracker_name,
        reid_model=None,
        reid_weights=weights,
        device="cuda:7",
        half=True,
        reid_preprocess="fast",
    )

    assert constructor_calls == []
    tracker.update(_detections(), _frame())

    assert constructor_calls == [
        {
            "weights": weights,
            "device": "cuda:7",
            "half": True,
            "preprocess_name": "fast",
        }
    ]
    assert len(model.calls) == 1


@pytest.mark.parametrize("tracker_name", REID_TRACKER_NAMES)
@pytest.mark.parametrize("frame_representation", ("canonical", "numpy"))
def test_reid_trackers_lazily_build_their_configured_encoder_from_the_full_spec(
    tracker_name: str,
    frame_representation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import boxmot.reid.factory as reid_factory

    encoder = _AppearanceEncoderSpy()
    factory_calls: list[ReIDEncoderSpec] = []
    spec = ReIDEncoderSpec(
        backend="native",
        artifact="models/reid.onnx",
        artifact_sha256="b" * 64,
        device="cpu",
        precision="fp32",
        options=(("batch_size", 7), ("embedding_dim", 3)),
        preprocessing="letterbox",
    )

    def create_encoder(received: ReIDEncoderSpec):
        factory_calls.append(received)
        return encoder

    monkeypatch.setattr(reid_factory, "create_reid_encoder", create_encoder)
    tracker = _tracker(tracker_name, reid_model=None)
    detections = _detections().with_masks(MaskBatch(torch.ones((2, 32, 48), dtype=torch.bool)))
    frame = _frame()
    input_frame = frame if frame_representation == "canonical" else np.full((32, 48, 3), (33, 22, 11), dtype=np.uint8)

    tracker.configure_reid(spec)
    assert factory_calls == []
    tracker.update(detections, input_frame)

    assert factory_calls == [spec]
    assert len(encoder.calls) == 1
    encoded_frames, encoded_detections = encoder.calls[0]
    assert encoded_detections == (detections,)
    assert len(encoded_frames) == 1
    encoded_frame = encoded_frames[0]
    assert isinstance(encoded_frame, Frame)
    encoded_frame.validate()
    assert encoded_frame.sample_id == detections.sample_id
    torch.testing.assert_close(encoded_frame.image, frame.image)
    if frame_representation == "canonical":
        assert encoded_frame is frame
    else:
        assert encoded_frame is not frame


@pytest.mark.parametrize("frame_representation", ("canonical", "numpy"))
def test_configured_encoder_rebuilds_numpy_rows_as_canonical_detections(
    frame_representation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import boxmot.reid.factory as reid_factory

    encoder = _AppearanceEncoderSpy()
    monkeypatch.setattr(reid_factory, "create_reid_encoder", lambda _spec: encoder)
    tracker = _tracker("botsort", reid_model=None)
    tracker.configure_reid(ReIDEncoderSpec("native", artifact="models/reid.onnx"))
    rows = np.array(
        [[4, 5, 20, 21, 0.8, 7], [24, 7, 40, 23, 0.6, 9]],
        dtype=np.float32,
    )
    frame = _frame()
    input_frame = frame if frame_representation == "canonical" else np.full((32, 48, 3), (33, 22, 11), dtype=np.uint8)

    tracker.update(rows, input_frame)

    encoded_frames, encoded_detections = encoder.calls[0]
    assert len(encoded_frames) == 1
    encoded_frame = encoded_frames[0]
    assert isinstance(encoded_frame, Frame)
    encoded_frame.validate()
    torch.testing.assert_close(encoded_frame.image, frame.image)
    if frame_representation == "canonical":
        assert encoded_frame is frame
    else:
        assert encoded_frame is not frame
    canonical = encoded_detections[0]
    assert canonical.sample_id == encoded_frame.sample_id
    torch.testing.assert_close(canonical.geometry.values, torch.from_numpy(rows[:, :4]))
    torch.testing.assert_close(canonical.scores, torch.from_numpy(rows[:, 4]))
    torch.testing.assert_close(canonical.class_ids, torch.tensor([7, 9], dtype=torch.int64))


@pytest.mark.parametrize("empty", (False, True))
def test_configured_encoder_is_not_built_when_no_live_inference_is_needed(
    empty: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import boxmot.reid.factory as reid_factory

    def reject_encoder_creation(_spec):
        raise AssertionError("The configured encoder should remain lazy.")

    monkeypatch.setattr(reid_factory, "create_reid_encoder", reject_encoder_creation)
    tracker = _tracker("botsort", reid_model=None)
    tracker.configure_reid(ReIDEncoderSpec("onnx", artifact="models/reid.onnx"))
    embeddings = None if empty else torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float32)

    tracker.update(_detections(embeddings=embeddings, empty=empty))


def test_reid_configuration_is_exclusive_and_must_precede_updates() -> None:
    spec = ReIDEncoderSpec("onnx", artifact="models/reid.onnx")
    configured = _tracker("botsort", reid_model=None)
    configured.configure_reid(spec)
    with pytest.raises(RuntimeError, match="already configured"):
        configured.configure_reid(spec)

    injected = _tracker(
        "botsort",
        reid_model=_ReIDModelSpy(np.ones((2, 3), dtype=np.float32)),
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        injected.configure_reid(spec)

    updated = _tracker("botsort", reid_model=None)
    updated.update(_detections(embeddings=torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float32)))
    with pytest.raises(RuntimeError, match="before the first update of a sequence"):
        updated.configure_reid(spec)
    updated.reset()
    updated.configure_reid(spec)


@pytest.mark.parametrize("tracker_name", OPTIONAL_REID_TRACKER_NAMES)
def test_optional_reid_trackers_do_not_run_model_when_embeddings_are_disabled(tracker_name: str) -> None:
    model = _ReIDModelSpy(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32))
    options = _tracker_options(tracker_name)
    options["use_embeddings"] = False
    tracker_class = get_tracker_class(tracker_name)
    tracker = tracker_class(**options, reid_model=model)

    tracker.update(_detections(), _frame())

    assert model.calls == []


def test_live_embeddings_are_normalized_and_keep_one_width_per_sequence() -> None:
    model = _ReIDModelSpy(np.array([[3, 4, 0], [0, 0, 2]], dtype=np.float32))
    tracker = _tracker("botsort", reid_model=model)
    geometry = _detections().geometry.values.numpy()

    features = tracker._resolve_input_embeddings(
        geometry=geometry,
        embeddings=None,
        frame=_frame(),
    )

    assert features is not None
    np.testing.assert_allclose(np.linalg.norm(features, axis=1), np.ones(2), atol=1e-6)
    with pytest.raises(ValueError, match="Embedding width changed from 3 to 4"):
        tracker._resolve_input_embeddings(
            geometry=geometry,
            embeddings=np.ones((2, 4), dtype=np.float32),
            frame=None,
        )


def test_live_embeddings_reject_zero_norm_features() -> None:
    model = _ReIDModelSpy(np.zeros((2, 3), dtype=np.float32))
    tracker = _tracker("botsort", reid_model=model)

    with pytest.raises(ValueError, match="zero-norm embedding"):
        tracker.update(_detections(), _frame())
