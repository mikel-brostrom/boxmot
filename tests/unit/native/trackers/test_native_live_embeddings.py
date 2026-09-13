"""Tracker-owned ReID fallback shared by native C++ tracker adapters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from boxmot.reid import EncoderRequirements, ReIDEncoderSpec
from boxmot.reid.specs import ReIDConfig
from boxmot.structures import Detections, Frame, Tracks
from boxmot.trackers.botsort.native import NativeBotSortTracker
from boxmot.trackers.occluboost.native import NativeOccluBoostTracker
from tests.unit.trackers._reid import INVALID_ENCODER_OUTPUTS, OutputEncoder, RecordingEncoder, invalid_output_encoder

from ._helpers import detections_from_rows, empty_native_batch, frame_from_bgr

NATIVE_REID_TRACKERS = (
    pytest.param(NativeBotSortTracker, id="botsort"),
    pytest.param(NativeOccluBoostTracker, id="occluboost"),
)


class _FakeLibrary:
    """Capture the typed buffers passed through the native tracker boundary."""

    def __init__(self) -> None:
        self.create_calls: list[dict[str, object]] = []
        self.update_calls: list[dict[str, np.ndarray | None]] = []
        self.reset_calls = 0
        self.destroy_calls = 0

    def create(self, cfg):
        self.create_calls.append(dict(cfg))
        return "handle"

    def reset(self, handle) -> None:
        assert handle == "handle"
        self.reset_calls += 1

    def update(
        self,
        handle,
        *,
        geometry,
        scores,
        class_ids,
        detection_indices,
        embeddings,
        image,
    ):
        assert handle == "handle"
        self.update_calls.append(
            {
                "geometry": geometry.copy(),
                "scores": scores.copy(),
                "class_ids": class_ids.copy(),
                "detection_indices": detection_indices.copy(),
                "embeddings": None if embeddings is None else embeddings.copy(),
                "image": None if image is None else image.copy(),
            }
        )
        return empty_native_batch(geometry.shape[1])

    def destroy(self, handle) -> None:
        assert handle == "handle"
        self.destroy_calls += 1


class _AppearanceEncoderSpy:
    """Canonical encoder double for full-spec lazy configuration."""

    embedding_dim = 3
    requirements = EncoderRequirements()

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Frame, ...], tuple[Detections, ...]]] = []

    def encode(self, frames, detections) -> list[torch.Tensor]:
        frame_batch = tuple(frames)
        detection_batch = tuple(detections)
        self.calls.append((frame_batch, detection_batch))
        features = torch.tensor([[3, 4, 0], [0, 0, 2]], dtype=torch.float32)
        return [features[: len(frame_detections)].clone() for frame_detections in detection_batch]


def _rows(geometry: str, *, empty: bool = False) -> np.ndarray:
    columns = 7 if geometry == "obb" else 6
    if empty:
        return np.empty((0, columns), dtype=np.float32)
    if geometry == "obb":
        return np.array(
            [[12, 13, 16, 14, 0.2, 0.8, 7], [32, 15, 16, 16, -0.3, 0.6, 9]],
            dtype=np.float32,
        )
    return np.array(
        [[4, 5, 20, 21, 0.8, 7], [24, 7, 40, 23, 0.6, 9]],
        dtype=np.float32,
    )


def _frame(*, sample_id: str = "sequence/000001") -> tuple[Frame, np.ndarray]:
    bgr = np.empty((32, 48, 3), dtype=np.uint8)
    bgr[:, :, 0] = 33
    bgr[:, :, 1] = 22
    bgr[:, :, 2] = 11
    return frame_from_bgr(bgr, sample_id=sample_id), bgr


def _tracker(tracker_class, library: _FakeLibrary, *, geometry: str = "aabb", **kwargs):
    return tracker_class(
        {"use_embeddings": True, "use_cmc": False},
        geometry=geometry,
        library=library,
        **kwargs,
    )


@pytest.mark.parametrize("tracker_class", NATIVE_REID_TRACKERS)
@pytest.mark.parametrize("geometry", ("aabb", "obb"))
@pytest.mark.parametrize("frame_representation", ("canonical", "numpy"))
def test_native_reid_trackers_encode_canonical_geometry_and_rgb_frames(
    tracker_class,
    geometry: str,
    frame_representation: str,
) -> None:
    library = _FakeLibrary()
    model = RecordingEncoder(np.array([[3, 4, 0], [0, 0, 2]], dtype=np.float32))
    tracker = _tracker(tracker_class, library, geometry=geometry, reid=model)
    detections = detections_from_rows(_rows(geometry), sample_id="sequence/000001")
    frame, expected_bgr = _frame()

    try:
        output = tracker.update(detections, frame if frame_representation == "canonical" else expected_bgr)
    finally:
        tracker.close()

    assert isinstance(output, Tracks)
    assert tracker.generates_embeddings is True
    assert len(model.calls) == 1
    frames, observations = model.calls[0]
    assert observations == (detections,)
    torch.testing.assert_close(frames[0].image, frame.image)
    assert frames[0].sample_id == detections.sample_id
    if frame_representation == "canonical":
        assert frames[0] is frame

    generated = library.update_calls[0]["embeddings"]
    assert generated is not None
    assert generated.dtype == np.float32
    assert generated.flags.c_contiguous
    np.testing.assert_allclose(generated, np.array([[0.6, 0.8, 0], [0, 0, 1]], dtype=np.float32))
    np.testing.assert_array_equal(library.update_calls[0]["image"], expected_bgr)


@pytest.mark.parametrize("tracker_class", NATIVE_REID_TRACKERS)
def test_native_reid_trackers_bypass_model_for_supplied_embeddings(tracker_class) -> None:
    library = _FakeLibrary()
    model = RecordingEncoder(np.ones((2, 3), dtype=np.float32))
    tracker = _tracker(tracker_class, library, reid=model)
    supplied = np.array([[7, 0, 0], [0, 5, 0]], dtype=np.float32)
    detections = detections_from_rows(_rows("aabb"), embeddings=supplied)

    try:
        tracker.update(detections)
    finally:
        tracker.close()

    assert model.calls == []
    np.testing.assert_array_equal(library.update_calls[0]["embeddings"], supplied)
    assert library.update_calls[0]["image"] is None


@pytest.mark.parametrize("tracker_class", NATIVE_REID_TRACKERS)
def test_native_reid_trackers_require_frame_only_for_nonempty_missing_embeddings(tracker_class) -> None:
    library = _FakeLibrary()
    model = RecordingEncoder(np.ones((2, 3), dtype=np.float32))
    tracker = _tracker(tracker_class, library, reid=model)

    try:
        with pytest.raises(ValueError, match="[Ff]rame"):
            tracker.update(detections_from_rows(_rows("aabb")))
    finally:
        tracker.close()

    assert model.calls == []
    assert library.update_calls == []


@pytest.mark.parametrize("tracker_class", NATIVE_REID_TRACKERS)
@pytest.mark.parametrize("geometry", ("aabb", "obb"))
@pytest.mark.parametrize("representation", ("canonical", "numpy"))
def test_native_reid_trackers_do_not_load_a_model_for_empty_batches(
    tracker_class,
    geometry: str,
    representation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import boxmot.reid.factory as reid_factory

    def reject_model_creation(_config):
        raise AssertionError("Empty batches must not initialize ReID.")

    monkeypatch.setattr(reid_factory, "create_reid_encoder", reject_model_creation)
    library = _FakeLibrary()
    tracker = _tracker(
        tracker_class,
        library,
        geometry=geometry,
        reid=ReIDConfig(model=Path("unused-reid.pt")),
    )
    rows = _rows(geometry, empty=True)
    detections = detections_from_rows(rows) if representation == "canonical" else rows

    try:
        output = tracker.update(detections)
    finally:
        tracker.close()

    assert len(library.update_calls) == 1
    embeddings = library.update_calls[0]["embeddings"]
    if embeddings is not None:
        assert embeddings.shape[0] == 0
        assert embeddings.dtype == np.float32
        assert embeddings.flags.c_contiguous
    if representation == "canonical":
        assert isinstance(output, Tracks)
        assert len(output) == 0
    else:
        assert type(output) is np.ndarray
        assert output.shape == (0, 9 if geometry == "obb" else 8)


@pytest.mark.parametrize("tracker_class", NATIVE_REID_TRACKERS)
def test_native_trackers_do_not_generate_embeddings_when_disabled(tracker_class) -> None:
    library = _FakeLibrary()
    model = RecordingEncoder(np.ones((2, 3), dtype=np.float32))
    tracker = tracker_class(
        {"use_embeddings": False, "use_cmc": False},
        library=library,
        reid=model,
    )

    try:
        tracker.update(detections_from_rows(_rows("aabb")))
    finally:
        tracker.close()

    assert tracker.generates_embeddings is False
    assert model.calls == []
    assert library.update_calls[0]["embeddings"] is None


@pytest.mark.parametrize("tracker_class", NATIVE_REID_TRACKERS)
def test_native_reid_trackers_build_the_configured_encoder_lazily(
    tracker_class,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import boxmot.reid.factory as reid_factory

    model = RecordingEncoder(np.array([[3, 4, 0], [0, 0, 2]], dtype=np.float32))
    constructor_calls: list[ReIDConfig] = []

    def create_encoder(config: ReIDConfig):
        constructor_calls.append(config)
        return model

    monkeypatch.setattr(reid_factory, "create_reid_encoder", create_encoder)
    library = _FakeLibrary()
    config = ReIDConfig(model=Path("custom-reid.pt"), device="cuda:7", precision="fp16", preprocessing="fast")
    tracker = _tracker(tracker_class, library, reid=config)
    frame, _ = _frame()

    assert constructor_calls == []
    try:
        tracker.update(detections_from_rows(_rows("aabb"), sample_id=frame.sample_id), frame)
    finally:
        tracker.close()

    assert constructor_calls == [config]
    assert len(model.calls) == 1


@pytest.mark.parametrize("tracker_class", NATIVE_REID_TRACKERS)
@pytest.mark.parametrize("geometry", ("aabb", "obb"))
@pytest.mark.parametrize("representation", ("canonical", "numpy"))
@pytest.mark.parametrize("frame_representation", ("canonical", "numpy"))
def test_native_reid_trackers_lazily_encode_canonical_and_numpy_inputs_from_full_spec(
    tracker_class,
    geometry: str,
    representation: str,
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
    library = _FakeLibrary()
    tracker = _tracker(tracker_class, library, geometry=geometry)
    rows = _rows(geometry)
    canonical = detections_from_rows(rows, sample_id="sequence/000001")
    input_detections = canonical if representation == "canonical" else rows
    frame, bgr = _frame()
    supplied = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

    tracker.configure_reid(spec)
    assert factory_calls == []
    try:
        tracker.update(detections_from_rows(_rows(geometry, empty=True)))
        tracker.update(detections_from_rows(rows, embeddings=supplied))
        assert factory_calls == []
        tracker.update(input_detections, frame if frame_representation == "canonical" else bgr)
    finally:
        tracker.close()

    assert factory_calls == [spec]
    assert len(encoder.calls) == 1
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
    encoded = encoded_detections[0]
    assert encoded.sample_id == encoded_frame.sample_id
    if representation == "canonical":
        assert encoded is canonical
    else:
        torch.testing.assert_close(encoded.geometry.values, torch.from_numpy(rows[:, :-2]))
        torch.testing.assert_close(encoded.scores, torch.from_numpy(rows[:, -2]))
        torch.testing.assert_close(encoded.class_ids, torch.tensor([7, 9], dtype=torch.int64))

    generated = library.update_calls[-1]["embeddings"]
    assert generated is not None
    assert generated.dtype == np.float32
    assert generated.flags.c_contiguous
    np.testing.assert_allclose(generated, np.array([[0.6, 0.8, 0], [0, 0, 1]], dtype=np.float32))
    np.testing.assert_array_equal(library.update_calls[-1]["image"], bgr)


@pytest.mark.parametrize("tracker_class", NATIVE_REID_TRACKERS)
def test_native_reid_configuration_must_precede_updates_but_reset_starts_a_new_sequence(tracker_class) -> None:
    spec = ReIDEncoderSpec("onnx", artifact="models/reid.onnx")
    first = _tracker(tracker_class, _FakeLibrary())
    first.configure_reid(spec)
    try:
        with pytest.raises(RuntimeError, match="already configured"):
            first.configure_reid(spec)
    finally:
        first.close()

    second = _tracker(tracker_class, _FakeLibrary())
    supplied = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    try:
        second.update(detections_from_rows(_rows("aabb"), embeddings=supplied))
        with pytest.raises(RuntimeError, match="before the first update of a sequence"):
            second.configure_reid(spec)
        second.reset()
        second.configure_reid(spec)
    finally:
        second.close()


@pytest.mark.parametrize("tracker_class", NATIVE_REID_TRACKERS)
@pytest.mark.parametrize("case", INVALID_ENCODER_OUTPUTS)
def test_native_encoder_output_contract_is_checked_before_native_update(tracker_class, case: str) -> None:
    encoder = invalid_output_encoder(case)
    library = _FakeLibrary()
    tracker = _tracker(tracker_class, library, reid=encoder)
    frame, _ = _frame()
    try:
        assert encoder.dimension_reads == 0
        with pytest.raises((TypeError, ValueError), match="ReID|embedding"):
            tracker.update(detections_from_rows(_rows("aabb"), sample_id=frame.sample_id), frame)
        assert encoder.calls == 1
        assert library.update_calls == []
        assert not tracker._has_updated
    finally:
        tracker.close()


@pytest.mark.parametrize("tracker_class", NATIVE_REID_TRACKERS)
def test_native_lazy_encoder_dimension_is_read_after_encoding_only(tracker_class) -> None:
    encoder = OutputEncoder(torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]]), defer_dimension=True)
    library = _FakeLibrary()
    tracker = _tracker(tracker_class, library, reid=encoder)
    frame, _ = _frame()
    try:
        tracker.update(detections_from_rows(_rows("aabb", empty=True), sample_id=frame.sample_id), frame)
        tracker.update(
            detections_from_rows(
                _rows("aabb"), embeddings=np.ones((2, 3), dtype=np.float32), sample_id=frame.sample_id
            ),
            frame,
        )
        assert encoder.calls == encoder.dimension_reads == 0
        tracker.update(detections_from_rows(_rows("aabb"), sample_id=frame.sample_id), frame)
        assert encoder.calls == encoder.dimension_reads == 1
        np.testing.assert_allclose(library.update_calls[-1]["embeddings"], [[0.6, 0.8, 0.0], [0.0, 0.0, 1.0]])
    finally:
        tracker.close()
