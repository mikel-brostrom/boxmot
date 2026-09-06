from __future__ import annotations

import ctypes
import importlib.util

import numpy as np
import pytest
import torch

from boxmot.native.trackers import _common
from boxmot.structures import Boxes, Detections, Frame, OrientedBoxes
from boxmot.trackers.box.bytetrack import native as native_bytetrack_module
from boxmot.trackers.box.bytetrack.native import NativeByteTrackTracker
from boxmot.trackers.box.sfsort.native import NativeSFSORTTracker
from boxmot.trackers.factory import create_tracker
from boxmot.trackers.protocols import TrackerRequirements
from boxmot.trackers.registry import supported_native_trackers
from boxmot.trackers.specs import TrackerSpec


def _detections(*, sample_id: str = "sample", class_id: int = 16_777_217) -> Detections:
    return Detections(
        geometry=Boxes(torch.tensor([[1.0, 2.0, 11.0, 22.0]], dtype=torch.float32)),
        scores=torch.tensor([0.9], dtype=torch.float32),
        class_ids=torch.tensor([class_id], dtype=torch.int64),
        sample_id=sample_id,
    )


def _buffers() -> dict[str, np.ndarray | None]:
    detections = _detections()
    return {
        "geometry": detections.geometry.values.numpy(),
        "scores": detections.scores.numpy(),
        "class_ids": detections.class_ids.numpy(),
        "detection_indices": np.arange(len(detections), dtype=np.int64),
        "embeddings": None,
        "image": None,
    }


def _empty_batch(columns: int = 4) -> _common.NativeTrackBatch:
    return _common.NativeTrackBatch(
        geometry=np.empty((0, columns), dtype=np.float32),
        scores=np.empty((0,), dtype=np.float32),
        track_ids=np.empty((0,), dtype=np.int64),
        class_ids=np.empty((0,), dtype=np.int64),
        detection_indices=np.empty((0,), dtype=np.int64),
    )


def test_typed_v2_call_preserves_int64_fields_and_frees_owned_result_once() -> None:
    calls = 0
    frees = 0
    keepalive: list[object] = []

    def update(_handle, batch_pointer, image_pointer, result_pointer) -> int:
        nonlocal calls
        calls += 1
        batch = ctypes.cast(batch_pointer, ctypes.POINTER(_common.CDetectionBatchV2)).contents
        assert batch.abi_version == 2
        assert batch.geometry_cols == 4
        assert batch.embedding_cols == 0
        assert batch.class_ids[0] == 16_777_217
        assert batch.detection_indices[0] == 0
        assert not image_pointer

        geometry = (ctypes.c_float * 8)(1.0, 2.0, 11.0, 22.0, 3.0, 4.0, 13.0, 24.0)
        scores = (ctypes.c_float * 2)(0.9, 0.8)
        track_ids = (ctypes.c_int64 * 2)(2**40 + 1, 2**40 + 2)
        class_ids = (ctypes.c_int64 * 2)(16_777_217, 16_777_219)
        detection_indices = (ctypes.c_int64 * 2)(0, -1)
        result = _common.CTrackBatchV2(
            abi_version=2,
            geometry=ctypes.cast(geometry, _common.FloatPointer),
            scores=ctypes.cast(scores, _common.FloatPointer),
            track_ids=ctypes.cast(track_ids, _common.Int64Pointer),
            class_ids=ctypes.cast(class_ids, _common.Int64Pointer),
            detection_indices=ctypes.cast(detection_indices, _common.Int64Pointer),
            rows=2,
            geometry_cols=4,
        )
        keepalive.extend((geometry, scores, track_ids, class_ids, detection_indices, result))
        destination = ctypes.cast(result_pointer, ctypes.POINTER(_common.CTrackBatchV2Pointer))
        destination[0] = ctypes.pointer(result)
        return 1

    def result_free(_result) -> None:
        nonlocal frees
        frees += 1

    tracks = _common.call_update_v2(
        update,
        result_free,
        handle=object(),
        **_buffers(),
        display_name="test",
        last_error=lambda: "failure",
    )

    assert calls == 1
    assert frees == 1
    assert tracks.track_ids.tolist() == [2**40 + 1, 2**40 + 2]
    assert tracks.class_ids.tolist() == [16_777_217, 16_777_219]
    assert tracks.detection_indices.tolist() == [0, -1]


def test_typed_v2_call_frees_result_when_output_validation_fails() -> None:
    calls = 0
    frees = 0
    result = _common.CTrackBatchV2(abi_version=999, rows=0, geometry_cols=4)

    def update(_handle, _batch, _image, result_pointer) -> int:
        nonlocal calls
        calls += 1
        ctypes.cast(result_pointer, ctypes.POINTER(_common.CTrackBatchV2Pointer))[0] = ctypes.pointer(result)
        return 1

    def result_free(_result) -> None:
        nonlocal frees
        frees += 1

    with pytest.raises(RuntimeError, match="unsupported ABI version"):
        _common.call_update_v2(
            update,
            result_free,
            handle=object(),
            **_buffers(),
            display_name="test",
            last_error=lambda: "failure",
        )
    assert calls == 1
    assert frees == 1


def test_low_level_binding_accepts_only_typed_contiguous_numpy_buffers() -> None:
    result = _common.CTrackBatchV2(abi_version=2, rows=0, geometry_cols=4)

    def update(_handle, _batch, _image, result_pointer) -> int:
        ctypes.cast(result_pointer, ctypes.POINTER(_common.CTrackBatchV2Pointer))[0] = ctypes.pointer(result)
        return 1

    buffers = _buffers()
    buffers["geometry"] = np.empty((1, 4), dtype=np.float64)
    with pytest.raises(TypeError, match="geometry must have dtype float32"):
        _common.call_update_v2(
            update,
            lambda _result: None,
            handle=object(),
            **buffers,
            display_name="test",
            last_error=lambda: "failure",
        )


class _FakeBinding:
    def __init__(self) -> None:
        self.image: np.ndarray | None = None
        self.geometry: np.ndarray | None = None
        self.scores: np.ndarray | None = None
        self.class_ids: np.ndarray | None = None
        self.detection_indices: np.ndarray | None = None

    def create(self, _cfg):
        return 1

    def destroy(self, _handle) -> None:
        return None

    def reset(self, _handle) -> None:
        return None

    def update(
        self,
        _handle,
        *,
        geometry,
        scores,
        class_ids,
        detection_indices,
        embeddings,
        image,
    ) -> _common.NativeTrackBatch:
        assert geometry.shape[1] == 4
        assert embeddings is None
        self.image = image
        self.geometry = geometry.copy()
        self.scores = scores.copy()
        self.class_ids = class_ids.copy()
        self.detection_indices = detection_indices.copy()
        return _empty_batch()


class _FakeObbBinding(_FakeBinding):
    def update(
        self,
        _handle,
        *,
        geometry,
        scores,
        class_ids,
        detection_indices,
        embeddings,
        image,
    ) -> _common.NativeTrackBatch:
        assert geometry.shape[1] == 5
        assert embeddings is None
        return _common.NativeTrackBatch(
            geometry=geometry.copy(),
            scores=scores.copy(),
            track_ids=np.array([9], dtype=np.int64),
            class_ids=class_ids.copy(),
            detection_indices=detection_indices.copy(),
        )


def test_frame_adapter_explicitly_converts_chw_rgb_to_hwc_bgr() -> None:
    library = _FakeBinding()
    tracker = NativeByteTrackTracker(geometry="aabb", library=library)
    frame = Frame(
        image=torch.tensor([[[1]], [[2]], [[3]]], dtype=torch.uint8),
        sample_id="sample",
    )
    tracker.update(_detections(), frame)
    assert library.image is not None
    assert library.image.reshape(-1).tolist() == [3, 2, 1]


def test_native_tracker_accepts_float64_numpy_rows_and_resets_generated_sample_ids() -> None:
    library = _FakeBinding()
    tracker = NativeByteTrackTracker(geometry="aabb", library=library)
    rows = np.array([[1, 2, 11, 22, 0.9, 16_777_217]], dtype=np.float64)

    first = tracker.update(rows)
    assert library.geometry is not None and library.geometry.dtype == np.float32
    assert library.scores is not None and library.scores.dtype == np.float32
    assert library.class_ids is not None and library.class_ids.tolist() == [16_777_217]
    assert library.detection_indices is not None and library.detection_indices.tolist() == [0]

    second = tracker.update(np.empty((0, 6), dtype=np.float64))
    tracker.reset()
    after_reset = tracker.update(np.empty((0, 6), dtype=np.float32))

    assert first.sample_id == "numpy:000000"
    assert second.sample_id == "numpy:000001"
    assert after_reset.sample_id == "numpy:000000"


def test_native_tracker_accepts_configured_obb7_rows_and_uses_frame_sample_id() -> None:
    tracker = NativeByteTrackTracker(geometry="obb", library=_FakeObbBinding())
    frame = Frame(
        image=torch.zeros((3, 8, 8), dtype=torch.uint8),
        sample_id="camera-1:000042",
    )
    rows = np.array([[20, 28, 32, 20, 0.1, 0.95, 2]], dtype=np.float64)

    tracks = tracker.update(rows, frame)

    assert tracks.sample_id == frame.sample_id
    assert tracks.is_obb
    assert tracks.class_ids.tolist() == [2]
    assert tracks.detection_indices.tolist() == [0]


def test_native_tracker_preserves_unwrapped_obb_angle_continuity() -> None:
    tracker = NativeByteTrackTracker(geometry="obb", library=_FakeObbBinding())

    def at_angle(angle: float, sample_id: str) -> Detections:
        return Detections(
            geometry=OrientedBoxes(torch.tensor([[20.0, 28.0, 32.0, 20.0, angle]], dtype=torch.float32)),
            scores=torch.tensor([0.95], dtype=torch.float32),
            class_ids=torch.tensor([2], dtype=torch.int64),
            sample_id=sample_id,
        )

    first = tracker.update(at_angle(3.1, "sequence/000001"))
    second = tracker.update(at_angle(-3.1, "sequence/000002"))
    assert float(second.geometry.values[0, 4]) > np.pi
    assert float(second.geometry.values[0, 4] - first.geometry.values[0, 4]) == pytest.approx(
        (2 * np.pi) - 6.2,
        abs=1e-5,
    )

    tracker.reset()
    after_reset = tracker.update(at_angle(-3.1, "sequence/000003"))
    assert float(after_reset.geometry.values[0, 4]) == pytest.approx(-3.1)


def test_native_sfsort_requires_frame() -> None:
    tracker = NativeSFSORTTracker(geometry="aabb", library=_FakeBinding())
    assert tracker.requirements.frame is True
    assert tracker.requirements.frame_dimensions_only is True
    assert tracker.requirements.frame_pixels is False
    with pytest.raises(ValueError, match="requires a frame"):
        tracker.update(_detections())


@pytest.mark.parametrize(
    "spec, message",
    [
        (TrackerSpec(name="bytetrack", backend="cpp", per_class=True), "per_class"),
        (
            TrackerSpec(name="bytetrack", backend="cpp", options=(("use_masks", True),)),
            "do not support masks",
        ),
        (
            TrackerSpec(name="botsort", backend="cpp", options=(("reid_weights", "model.pt"),)),
            "precomputed embeddings",
        ),
        (
            TrackerSpec(name="botsort", backend="cpp", options=(("removed_stracks_buffer", 20),)),
            "does not implement",
        ),
        (
            TrackerSpec(name="occluboost", backend="cpp", options=(("adaptive_kf", True),)),
            "does not implement",
        ),
        (TrackerSpec(name="strongsort", backend="cpp"), "unavailable"),
    ],
)
def test_native_factory_rejects_unsupported_modes(spec: TrackerSpec, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        create_tracker(spec)


def test_native_factory_registry_is_owned_by_trackers() -> None:
    assert supported_native_trackers() == (
        "botsort",
        "bytetrack",
        "occluboost",
        "ocsort",
        "sfsort",
    )
    assert importlib.util.find_spec("boxmot.native.registry") is None


def test_public_factory_dispatches_structured_native_spec(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class StubNativeTracker:
        def __init__(self, options, *, geometry):
            captured.update(options=options, geometry=geometry)
            self.requirements = TrackerRequirements()

    monkeypatch.setattr(native_bytetrack_module, "NativeByteTrackTracker", StubNativeTracker)
    tracker = create_tracker(
        TrackerSpec(
            name="bytetrack",
            backend="cpp",
            geometry="obb",
            options=(("track_buffer", 42),),
        )
    )
    assert isinstance(tracker, StubNativeTracker)
    assert captured == {"options": {"track_buffer": 42}, "geometry": "obb"}
