"""Tracker-owned ReID fallback shared by native C++ tracker adapters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from boxmot.reid import EncoderRequirements, ReIDEncoderSpec
from boxmot.structures import Detections, Frame, Tracks
from boxmot.trackers.botsort.native import NativeBotSortTracker
from boxmot.trackers.occluboost.native import NativeOccluBoostTracker

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


class _ReIDModelSpy:
    """Record calls through the established ``get_features`` runtime API."""

    def __init__(self, features: np.ndarray) -> None:
        self.features = np.asarray(features, dtype=np.float32)
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []

    def get_features(self, geometry: np.ndarray, image: np.ndarray) -> np.ndarray:
        self.calls.append((geometry.copy(), image.copy()))
        return self.features[: len(geometry)].copy()


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
def test_native_reid_trackers_generate_missing_embeddings_from_geometry_and_bgr_frame(
    tracker_class,
    geometry: str,
    frame_representation: str,
) -> None:
    library = _FakeLibrary()
    model = _ReIDModelSpy(np.array([[3, 4, 0], [0, 0, 2]], dtype=np.float32))
    tracker = _tracker(tracker_class, library, geometry=geometry, reid_model=model)
    detections = detections_from_rows(_rows(geometry), sample_id="sequence/000001")
    frame, expected_bgr = _frame()

    try:
        output = tracker.update(detections, frame if frame_representation == "canonical" else expected_bgr)
    finally:
        tracker.close()

    assert isinstance(output, Tracks)
    assert tracker.generates_embeddings is True
    assert len(model.calls) == 1
    received_geometry, received_image = model.calls[0]
    np.testing.assert_array_equal(received_geometry, detections.geometry.values.numpy())
    np.testing.assert_array_equal(received_image, expected_bgr)
    assert received_image.flags.c_contiguous

    generated = library.update_calls[0]["embeddings"]
    assert generated is not None
    assert generated.dtype == np.float32
    assert generated.flags.c_contiguous
    np.testing.assert_allclose(generated, np.array([[0.6, 0.8, 0], [0, 0, 1]], dtype=np.float32))
    np.testing.assert_array_equal(library.update_calls[0]["image"], expected_bgr)


@pytest.mark.parametrize("tracker_class", NATIVE_REID_TRACKERS)
def test_native_reid_trackers_bypass_model_for_supplied_embeddings(tracker_class) -> None:
    library = _FakeLibrary()
    model = _ReIDModelSpy(np.ones((2, 3), dtype=np.float32))
    tracker = _tracker(tracker_class, library, reid_model=model)
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
    model = _ReIDModelSpy(np.ones((2, 3), dtype=np.float32))
    tracker = _tracker(tracker_class, library, reid_model=model)

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
    import boxmot.reid.core as reid_core

    def reject_model_creation(**_kwargs):
        raise AssertionError("Empty batches must not initialize ReID.")

    monkeypatch.setattr(reid_core, "ReID", reject_model_creation)
    library = _FakeLibrary()
    tracker = _tracker(
        tracker_class,
        library,
        geometry=geometry,
        reid_weights=Path("unused-reid.pt"),
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
    model = _ReIDModelSpy(np.ones((2, 3), dtype=np.float32))
    tracker = tracker_class(
        {"use_embeddings": False, "use_cmc": False},
        library=library,
        reid_model=model,
    )

    try:
        tracker.update(detections_from_rows(_rows("aabb")))
    finally:
        tracker.close()

    assert tracker.generates_embeddings is False
    assert model.calls == []
    assert library.update_calls[0]["embeddings"] is None


@pytest.mark.parametrize("tracker_class", NATIVE_REID_TRACKERS)
def test_native_reid_trackers_build_raw_model_lazily_with_shared_options(
    tracker_class,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import boxmot.reid.core as reid_core

    model = _ReIDModelSpy(np.array([[3, 4, 0], [0, 0, 2]], dtype=np.float32))
    constructor_calls: list[dict[str, object]] = []

    class _ReIDRuntime:
        def __init__(self, **kwargs: object) -> None:
            constructor_calls.append(kwargs)
            self.model = model

    monkeypatch.setattr(reid_core, "ReID", _ReIDRuntime)
    library = _FakeLibrary()
    weights = Path("custom-reid.pt")
    tracker = _tracker(
        tracker_class,
        library,
        reid_weights=weights,
        device="cuda:7",
        half=True,
        reid_preprocess="fast",
    )
    frame, _ = _frame()

    assert constructor_calls == []
    try:
        tracker.update(detections_from_rows(_rows("aabb"), sample_id=frame.sample_id), frame)
    finally:
        tracker.close()

    assert constructor_calls == [
        {
            "weights": weights,
            "device": "cuda:7",
            "half": True,
            "preprocess_name": "fast",
        }
    ]
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
