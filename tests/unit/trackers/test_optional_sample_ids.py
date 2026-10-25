"""Direct tracking resolves optional detection IDs without mutating caller inputs."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from boxmot import BotSort, BotSortConfig, ByteTrack, EagerMot, StrongSort, StrongSortConfig
from boxmot.structures import Boxes, Boxes3D, CameraModel, Detections, Detections3D, Frame, Tracks
from boxmot.trackers.bytetrack.native import NativeByteTrackTracker
from tests.unit.native.test_native_abi_v2 import _FakeBinding
from tests.unit.trackers._reid import RecordingEncoder


def _inputs() -> tuple[Detections, Detections3D, CameraModel]:
    """Construct the public EagerMot example with both sample arguments omitted."""
    detections = Detections(
        geometry=Boxes(torch.tensor([[78, 39, 122, 62]], dtype=torch.float32)),
        scores=torch.tensor([0.95]),
        class_ids=torch.tensor([0]),
    )
    spatial = Detections3D(
        geometry=Boxes3D(torch.tensor([[0, 1, 10, 0, 4, 2, 2]], dtype=torch.float32)),
        scores=torch.tensor([0.9]),
        class_ids=torch.tensor([0]),
    )
    camera = CameraModel(
        projection=torch.tensor([[100, 0, 100, 0], [0, 100, 50, 0], [0, 0, 1, 0]], dtype=torch.float32),
        image_size=(100, 200),
    )
    return detections, spatial, camera


def _frame(sample_id: str = "camera:42") -> Frame:
    """Provide a canonical frame with identity supplied by the source."""
    return Frame(torch.zeros((3, 100, 200), dtype=torch.uint8), sample_id=sample_id)


def test_eagermot_assigns_sample_ids_and_preserves_track_identity_across_updates() -> None:
    """Reusing anonymous observations advances sample identity without losing tracks."""
    tracker = EagerMot()
    detections, spatial, camera = _inputs()
    outputs = [tracker.update(detections, detections_3d=spatial, camera=camera) for _ in range(3)]

    for index, output in enumerate(outputs):
        expected_id = f"tracker:{index}"
        assert output.sample_id == output.image_tracks.sample_id == output.spatial_tracks.sample_id == expected_id
        assert len(output.image_tracks) == len(output.spatial_tracks) == 1
        torch.testing.assert_close(output.image_tracks.track_ids, outputs[0].image_tracks.track_ids)
        torch.testing.assert_close(output.spatial_tracks.track_ids, outputs[0].image_tracks.track_ids)
    assert detections.sample_id is spatial.sample_id is None

    tracker.reset()
    restarted = tracker.update(detections, detections_3d=spatial, camera=camera)
    assert restarted.sample_id == "tracker:0"


@pytest.mark.parametrize("empty_sensor", ("2d", "3d", "both"))
def test_eagermot_assigns_sample_identity_to_empty_sensor_batches(empty_sensor: str) -> None:
    """Empty sensor results use the same identity resolution as populated inputs."""
    detections, spatial, camera = _inputs()
    indices = torch.empty(0, dtype=torch.int64)
    if empty_sensor in ("2d", "both"):
        detections = detections.select(indices)
    if empty_sensor in ("3d", "both"):
        spatial = spatial.select(indices)

    output = EagerMot().update(detections, detections_3d=spatial, camera=camera)

    assert output.image_tracks.sample_id == output.spatial_tracks.sample_id == "tracker:0"
    assert detections.sample_id is spatial.sample_id is None


@pytest.mark.parametrize("identity_source", ("2d", "3d", "frame", "all"))
def test_eagermot_infers_a_shared_sample_id_from_supplied_inputs(identity_source: str) -> None:
    """An authored identity takes precedence over tracker-generated identities."""
    detections, spatial, camera = _inputs()
    if identity_source in ("2d", "all"):
        detections = replace(detections, sample_id="camera:42")
    if identity_source in ("3d", "all"):
        spatial = replace(spatial, sample_id="camera:42")
    frame = _frame() if identity_source in ("frame", "all") else None
    original_ids = (detections.sample_id, spatial.sample_id)

    output = EagerMot().update(detections, frame, detections_3d=spatial, camera=camera)

    assert output.image_tracks.sample_id == output.spatial_tracks.sample_id == "camera:42"
    assert (detections.sample_id, spatial.sample_id) == original_ids


@pytest.mark.parametrize(
    "image_id,spatial_id,frame_id",
    (("first", "second", None), ("first", None, "second"), (None, "first", "second")),
)
def test_eagermot_rejects_conflicting_explicit_ids_before_advancing_state(
    image_id: str | None, spatial_id: str | None, frame_id: str | None
) -> None:
    """Making IDs optional must preserve checks against mixing different samples."""
    tracker = EagerMot()
    detections, spatial, camera = _inputs()
    with pytest.raises(ValueError, match="same sample"):
        tracker.update(
            replace(detections, sample_id=image_id),
            _frame(frame_id) if frame_id is not None else None,
            detections_3d=replace(spatial, sample_id=spatial_id),
            camera=camera,
        )

    assert tracker.frame_count == 0
    assert not tracker._first_frame_processed
    assert tracker._tracks == []
    output = tracker.update(detections, detections_3d=spatial, camera=camera)
    assert output.sample_id == "tracker:0"


@pytest.mark.parametrize("backend", ("python", "native"))
@pytest.mark.parametrize("frame_kind", ("none", "numpy", "canonical"))
def test_box_trackers_resolve_missing_ids_for_canonical_detection_inputs(backend: str, frame_kind: str) -> None:
    """Both box backends return identified canonical outputs for anonymous inputs."""
    tracker = ByteTrack() if backend == "python" else NativeByteTrackTracker(library=_FakeBinding())
    detections, _, _ = _inputs()
    frame = _frame() if frame_kind == "canonical" else None
    if frame_kind == "numpy":
        frame = np.zeros((100, 200, 3), dtype=np.uint8)

    for index in range(2):
        output = tracker.update(detections, frame)
        assert isinstance(output, Tracks)
        assert output.sample_id == ("camera:42" if frame_kind == "canonical" else f"tracker:{index}")
    assert detections.sample_id is None

    tracker.reset()
    assert tracker.update(detections).sample_id == "tracker:0"


@pytest.mark.parametrize("per_class", (False, True))
def test_strongsort_advances_sample_ids_independently_of_kernel_frame_count(per_class: bool) -> None:
    """Sample identity advances once per public call in either dispatch mode."""
    tracker = StrongSort(config=StrongSortConfig(n_init=1, use_cmc=False), per_class=per_class)
    detections, _, _ = _inputs()
    detections = detections.with_embeddings(torch.tensor([[1.0, 0.0, 0.0]]))

    first = tracker.update(detections)
    second = tracker.update(detections)

    assert first.sample_id == "tracker:0"
    assert second.sample_id == "tracker:1"
    assert detections.sample_id is None

    tracker.reset()
    assert tracker.update(detections).sample_id == "tracker:0"


@pytest.mark.parametrize("backend", ("python", "native"))
def test_sample_ids_count_packed_explicit_and_empty_updates(backend: str) -> None:
    """Changing input representation does not reuse an earlier sample identity."""
    tracker = ByteTrack() if backend == "python" else NativeByteTrackTracker(library=_FakeBinding())
    detections, _, _ = _inputs()
    empty = detections.select(torch.empty(0, dtype=torch.int64))

    tracker.update(detections.to_aabb_rows().numpy())
    assert tracker.update(empty).sample_id == "tracker:1"
    assert tracker.update(replace(empty, sample_id="camera:42")).sample_id == "camera:42"
    tracker.update(np.empty((0, 6), dtype=np.float32))
    assert tracker.update(empty).sample_id == "tracker:4"
    assert detections.sample_id is empty.sample_id is None


@pytest.mark.parametrize("frame_kind", ("numpy", "canonical"))
def test_live_reid_receives_resolved_sample_ids(frame_kind: str) -> None:
    """Live appearance encoding sees aligned identified inputs before tracking."""
    encoder = RecordingEncoder(np.array([[1, 0, 0]], dtype=np.float32))
    tracker = BotSort(config=BotSortConfig(use_cmc=False), reid=encoder)
    detections, _, _ = _inputs()
    frame = _frame() if frame_kind == "canonical" else np.zeros((100, 200, 3), dtype=np.uint8)

    output = tracker.update(detections, frame)

    assert len(encoder.calls) == 1
    frames, observations = encoder.calls[0]
    expected_id = "camera:42" if frame_kind == "canonical" else "tracker:0"
    assert frames[0].sample_id == observations[0].sample_id == output.sample_id == expected_id
    assert observations[0].geometry is detections.geometry
    assert detections.sample_id is None
