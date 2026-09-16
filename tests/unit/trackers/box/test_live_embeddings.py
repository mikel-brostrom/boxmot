"""Live appearance extraction shared by Python ReID-enabled trackers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from boxmot.components.timing import timing_event_sink
from boxmot.reid import EncoderRequirements, ReIDEncoderSpec
from boxmot.reid.adapters import RuntimeAppearanceEncoder
from boxmot.reid.protocols import AppearanceEncoder
from boxmot.reid.specs import ReIDConfig
from boxmot.structures import Boxes, Detections, Frame, MaskBatch, OrientedBoxes
from boxmot.trackers.common.config import get_tracker_config_class
from boxmot.trackers.common.registry import TRACKER_DEFINITIONS, get_tracker_class
from tests.unit.trackers._reid import INVALID_ENCODER_OUTPUTS, OutputEncoder, RecordingEncoder, invalid_output_encoder

REID_TRACKER_NAMES = tuple(
    name for name, definition in TRACKER_DEFINITIONS.items() if definition.capabilities.accepts_embeddings
)
OPTIONAL_REID_TRACKER_NAMES = tuple(
    name for name in REID_TRACKER_NAMES if not TRACKER_DEFINITIONS[name].capabilities.requires_embeddings
)


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
                "instant_confirm_thresh": 0.2,
                "new_track_thresh": 0.2,
            },
            "strongsort": {"min_conf": 0.2, "n_init": 1},
        }[tracker_name]
    )
    return common


def _tracker(tracker_name: str, *, reid: ReIDConfig | AppearanceEncoder | None, **kwargs: object):
    tracker_class = get_tracker_class(tracker_name)
    return tracker_class(
        config=get_tracker_config_class(tracker_name)(**_tracker_options(tracker_name)),
        reid=reid,
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
def test_reid_trackers_encode_canonical_geometry_and_rgb_frames(
    tracker_name: str,
    geometry: str,
    frame_representation: str,
) -> None:
    expected_features = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    model = RecordingEncoder(expected_features)
    tracker = _tracker(tracker_name, reid=model, is_obb=geometry == "obb")
    detections = _detections(geometry=geometry)
    frame = _frame() if frame_representation == "canonical" else np.full((32, 48, 3), (33, 22, 11), dtype=np.uint8)

    tracker.update(detections, frame)

    assert len(model.calls) == 1
    frames, observations = model.calls[0]
    assert observations == (detections,)
    torch.testing.assert_close(frames[0].image, _frame().image)
    assert frames[0].sample_id == detections.sample_id
    if frame_representation == "canonical":
        assert frames[0] is frame


def test_direct_live_reid_emits_component_timing_phases() -> None:
    runtime = SimpleNamespace(
        get_features=lambda boxes, _image: np.eye(len(boxes), 3, dtype=np.float32),
    )
    encoder = RuntimeAppearanceEncoder(
        ReIDEncoderSpec("native", options=(("embedding_dim", 3), ("image_size", (32, 16)))),
        runtime,
    )
    tracker = _tracker("botsort", reid=encoder)
    events = []

    with timing_event_sink(events.append):
        tracker.update(_detections(), _frame())

    assert {(event.component, event.phase) for event in events} == {
        ("reid", "preprocess"),
        ("reid", "process"),
        ("reid", "postprocess"),
    }


def test_canonical_live_reid_skips_bgr_conversion_when_tracking_needs_no_pixels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = RecordingEncoder(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32))
    tracker = _tracker("botsort", reid=model)
    original = tracker._frame_to_bgr
    converted_images: list[np.ndarray] = []

    def record_conversion(frame: Frame) -> np.ndarray:
        converted = original(frame)
        converted_images.append(converted)
        return converted

    monkeypatch.setattr(tracker, "_frame_to_bgr", record_conversion)

    tracker.update(_detections(), _frame())

    assert converted_images == []
    assert len(model.calls) == 1


@pytest.mark.parametrize("tracker_name", REID_TRACKER_NAMES)
def test_reid_trackers_bypass_model_when_embeddings_are_precomputed(tracker_name: str) -> None:
    model = RecordingEncoder(np.array([[1, 0, 0]], dtype=np.float32))
    tracker = _tracker(tracker_name, reid=model)
    supplied = torch.tensor([[0, 0, 1], [1, 0, 0]], dtype=torch.float32)

    tracker.update(_detections(embeddings=supplied), _frame())

    assert model.calls == []


@pytest.mark.parametrize("tracker_name", REID_TRACKER_NAMES)
def test_reid_trackers_require_frame_only_when_live_embeddings_are_needed(tracker_name: str) -> None:
    model = RecordingEncoder(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32))
    tracker = _tracker(tracker_name, reid=model)

    with pytest.raises(ValueError, match="[Ff]rame"):
        tracker.update(_detections())

    assert model.calls == []


@pytest.mark.parametrize("tracker_name", REID_TRACKER_NAMES)
def test_reid_trackers_skip_live_model_for_empty_batches(tracker_name: str) -> None:
    model = RecordingEncoder(np.empty((0, 3), dtype=np.float32))
    tracker = _tracker(tracker_name, reid=model)

    tracks = tracker.update(_detections(empty=True), _frame())

    assert len(tracks) == 0
    assert model.calls == []


@pytest.mark.parametrize("tracker_name", REID_TRACKER_NAMES)
def test_reid_trackers_build_the_configured_encoder_lazily(
    tracker_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import boxmot.reid.factory as reid_factory

    model = RecordingEncoder(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32))
    constructor_calls: list[ReIDConfig] = []

    def create_encoder(config: ReIDConfig):
        constructor_calls.append(config)
        return model

    monkeypatch.setattr(reid_factory, "create_reid_encoder", create_encoder)
    config = ReIDConfig(model=Path("custom-reid.pt"), device="cuda:7", precision="fp16", preprocessing="fast")
    tracker = _tracker(tracker_name, reid=config)

    assert constructor_calls == []
    tracker.update(_detections(), _frame())
    assert constructor_calls == [config]
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
    tracker = _tracker(tracker_name, reid=None)
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
    encoder.requirements = EncoderRequirements()
    monkeypatch.setattr(reid_factory, "create_reid_encoder", lambda _spec: encoder)
    tracker = _tracker("botsort", reid=None)
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


@pytest.mark.parametrize("tracker_name", REID_TRACKER_NAMES)
@pytest.mark.parametrize("injected", [False, True])
def test_live_encoder_required_masks_are_checked_before_encoding(
    tracker_name: str, injected: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The encoder's mandatory channels apply even when tracking uses only boxes."""
    import boxmot.reid.factory as reid_factory

    encoder = _AppearanceEncoderSpy()
    monkeypatch.setattr(reid_factory, "create_reid_encoder", lambda _spec: encoder)
    tracker = _tracker(tracker_name, reid=encoder if injected else None)
    if not injected:
        tracker.configure_reid(ReIDEncoderSpec("native", artifact="models/reid.onnx"))

    with pytest.raises(ValueError, match="live ReID requires full-frame detection masks"):
        tracker.update(_detections(), _frame())

    assert encoder.calls == []
    assert not tracker._has_updated


@pytest.mark.parametrize("tracker_name", REID_TRACKER_NAMES)
def test_mask_aware_live_encoder_accepts_empty_batches_without_inference(
    tracker_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty provider output advances tracking before and after lazy ReID creation."""
    import boxmot.reid.factory as reid_factory

    encoder = _AppearanceEncoderSpy()
    constructed = []

    def create_encoder(spec: ReIDEncoderSpec) -> _AppearanceEncoderSpy:
        constructed.append(spec)
        return encoder

    monkeypatch.setattr(reid_factory, "create_reid_encoder", create_encoder)
    tracker = _tracker(tracker_name, reid=None)
    tracker.configure_reid(ReIDEncoderSpec("native", artifact="models/reid.onnx"))
    empty = _detections(empty=True).with_masks(MaskBatch(torch.empty((0, 32, 48), dtype=torch.bool)))

    tracker.update(empty, _frame())
    assert tracker._has_updated
    assert constructed == []

    tracker.update(_detections().with_masks(MaskBatch(torch.ones((2, 32, 48), dtype=torch.bool))), _frame())
    tracker.update(empty, _frame())
    assert len(constructed) == 1
    assert len(encoder.calls) == 1


@pytest.mark.parametrize("empty", (False, True))
def test_configured_encoder_is_not_built_when_no_live_inference_is_needed(
    empty: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import boxmot.reid.factory as reid_factory

    def reject_encoder_creation(_spec):
        raise AssertionError("The configured encoder should remain lazy.")

    monkeypatch.setattr(reid_factory, "create_reid_encoder", reject_encoder_creation)
    tracker = _tracker("botsort", reid=None)
    tracker.configure_reid(ReIDEncoderSpec("onnx", artifact="models/reid.onnx"))
    embeddings = None if empty else torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float32)

    tracker.update(_detections(embeddings=embeddings, empty=empty))


def test_reid_configuration_is_exclusive_and_must_precede_updates() -> None:
    spec = ReIDEncoderSpec("onnx", artifact="models/reid.onnx")
    configured = _tracker("botsort", reid=None)
    configured.configure_reid(spec)
    with pytest.raises(RuntimeError, match="already configured"):
        configured.configure_reid(spec)

    injected = _tracker(
        "botsort",
        reid=RecordingEncoder(np.ones((2, 3), dtype=np.float32)),
    )
    with pytest.raises(RuntimeError, match="already configured"):
        injected.configure_reid(spec)

    updated = _tracker("botsort", reid=None)
    updated.update(_detections(embeddings=torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float32)))
    with pytest.raises(RuntimeError, match="before the first update of a sequence"):
        updated.configure_reid(spec)
    updated.reset()
    updated.configure_reid(spec)


@pytest.mark.parametrize("tracker_name", OPTIONAL_REID_TRACKER_NAMES)
def test_optional_reid_trackers_do_not_run_model_when_embeddings_are_disabled(tracker_name: str) -> None:
    model = RecordingEncoder(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32))
    options = _tracker_options(tracker_name)
    options["use_embeddings"] = False
    tracker_class = get_tracker_class(tracker_name)
    tracker = tracker_class(config=get_tracker_config_class(tracker_name)(**options), reid=model)

    tracker.update(_detections(), _frame())

    assert model.calls == []


def test_live_embeddings_are_normalized_and_keep_one_width_per_sequence() -> None:
    model = RecordingEncoder(np.array([[3, 4, 0], [0, 0, 2]], dtype=np.float32))
    tracker = _tracker("botsort", reid=model)
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
    model = RecordingEncoder(np.zeros((2, 3), dtype=np.float32))
    tracker = _tracker("botsort", reid=model)

    with pytest.raises(ValueError, match="zero-norm embedding"):
        tracker.update(_detections(), _frame())


@pytest.mark.parametrize("tracker_name", REID_TRACKER_NAMES)
@pytest.mark.parametrize("case", INVALID_ENCODER_OUTPUTS)
def test_live_encoder_output_contract_is_checked_before_tracking(tracker_name: str, case: str) -> None:
    encoder = invalid_output_encoder(case)
    tracker = _tracker(tracker_name, reid=encoder)
    assert encoder.dimension_reads == 0
    with pytest.raises((TypeError, ValueError), match="ReID|embedding"):
        tracker.update(_detections(), _frame())
    assert encoder.calls == 1
    assert tracker.frame_count == 0
    assert not tracker._has_updated


@pytest.mark.parametrize("tracker_name", REID_TRACKER_NAMES)
def test_lazy_encoder_dimension_is_read_after_encoding_only(tracker_name: str) -> None:
    encoder = OutputEncoder(torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]]), defer_dimension=True)
    tracker = _tracker(tracker_name, reid=encoder)
    tracker.update(_detections(empty=True), _frame())
    tracker.update(_detections(embeddings=torch.ones((2, 3))), _frame())
    assert encoder.calls == encoder.dimension_reads == 0
    tracker.update(_detections(), _frame())
    assert encoder.calls == encoder.dimension_reads == 1
    assert tracker.last_emb_size == 3
