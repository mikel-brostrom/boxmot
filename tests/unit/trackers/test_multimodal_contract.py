"""Shared validation for calibrated tracking with independent sensor inputs."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from boxmot import ByteTrack
from boxmot.structures import (
    Boxes,
    Boxes3D,
    CameraModel,
    Detections,
    Detections3D,
    Frame,
    GeometryKind,
    MultimodalTracks,
    Tracks,
    Tracks3D,
)
from boxmot.trackers.common.base import BaseTracker
from boxmot.trackers.common.specs import TrackerCapabilities, TrackerFamily


class _FusionTracker(BaseTracker):
    """Exercise dispatch and validation without introducing a tracking algorithm."""

    capabilities = TrackerCapabilities(
        family=TrackerFamily.MULTIMODAL,
        geometry_kinds=frozenset({GeometryKind.AABB}),
        requires_detections_3d=True,
        accepts_detections_3d=True,
        requires_camera=True,
        accepts_camera=True,
    )
    _requires_detections_3d = True
    _requires_camera = True
    uses_frame_dimensions_for_association = False

    def _track_detections(self, dets, img, embs=None, masks=None):
        raise AssertionError("The independent sensor inputs must not enter a packed 2D kernel.")

    def _track_multimodal(self, detections, detections_3d, camera, frame):
        self.frame_count += 1
        self.received = (detections, detections_3d, camera, frame)
        return MultimodalTracks(
            Tracks(
                geometry=detections.geometry,
                track_ids=torch.arange(len(detections)),
                scores=detections.scores,
                class_ids=detections.class_ids,
                detection_indices=torch.arange(len(detections)),
                sample_id=detections.sample_id,
            ),
            Tracks3D(
                geometry=detections_3d.geometry,
                track_ids=torch.arange(len(detections_3d)) + len(detections),
                scores=detections_3d.scores,
                class_ids=detections_3d.class_ids,
                detection_indices=torch.arange(len(detections_3d)),
                sample_id=detections_3d.sample_id,
            ),
        )


def _inputs() -> tuple[Detections, Detections3D, CameraModel]:
    """Use different sensor counts and exact int64 classes beyond float32 precision."""
    detections = Detections(
        Boxes(torch.tensor([[10.0, 10.0, 20.0, 30.0], [30.0, 10.0, 40.0, 30.0]])),
        torch.tensor([0.9, 0.8]),
        torch.tensor([2**40, 3]),
        "camera:0",
    )
    spatial = Detections3D(
        Boxes3D(torch.tensor([[0.0, 1.0, -5.0, 0.0, 4.0, 2.0, 1.0]])),
        torch.tensor([0.9]),
        torch.tensor([2**50]),
        "camera:0",
    )
    camera = CameraModel(
        torch.tensor([[100.0, 0.0, 32.0, 0.0], [0.0, 100.0, 24.0, 0.0], [0.0, 0.0, 1.0, 0.0]]),
        (48, 64),
    )
    return detections, spatial, camera


@pytest.mark.parametrize("per_class", (False, True))
def test_inherited_update_preserves_independent_modalities_and_exact_class_ids(per_class: bool) -> None:
    tracker = _FusionTracker(per_class=per_class)
    detections, spatial, camera = _inputs()
    output = tracker.update(detections, detections_3d=spatial, camera=camera)

    assert _FusionTracker.update is BaseTracker.update
    assert len(output.image_tracks) == 2
    assert len(output.spatial_tracks) == 1
    assert output.image_tracks.class_ids.tolist() == [2**40, 3]
    assert output.spatial_tracks.class_ids.tolist() == [2**50]
    assert tracker.frame_count == 1
    assert tracker.received[0] is detections and tracker.received[1] is spatial
    assert tracker._canonical_to_kernel_class_id == {}


@pytest.mark.parametrize("empty_sensor", ("2d", "3d", "both"))
def test_completed_empty_sensor_batches_are_independent(empty_sensor: str) -> None:
    detections, spatial, camera = _inputs()
    if empty_sensor in ("2d", "both"):
        detections = detections.select(torch.empty(0, dtype=torch.int64))
    if empty_sensor in ("3d", "both"):
        spatial = spatial.select(torch.empty(0, dtype=torch.int64))
    output = _FusionTracker().update(detections, detections_3d=spatial, camera=camera)
    assert len(output.image_tracks) == len(detections)
    assert len(output.spatial_tracks) == len(spatial)


@pytest.mark.parametrize("missing", ("detections_3d", "camera"))
def test_required_sensor_inputs_are_not_silently_treated_as_empty(missing: str) -> None:
    tracker = _FusionTracker()
    detections, spatial, camera = _inputs()
    kwargs = {"detections_3d": spatial, "camera": camera}
    del kwargs[missing]
    with pytest.raises(ValueError, match=f"requires {missing}"):
        tracker.update(detections, **kwargs)
    assert tracker.frame_count == 0 and not tracker._first_frame_processed


def test_other_trackers_reject_3d_side_inputs_and_fusion_rejects_packed_rows() -> None:
    detections, spatial, camera = _inputs()
    with pytest.raises(ValueError, match="does not accept detections_3d"):
        ByteTrack().update(detections, detections_3d=spatial, camera=camera)
    with pytest.raises(ValueError, match="does not accept camera"):
        ByteTrack().update(detections, camera=camera)
    with pytest.raises(TypeError, match="canonical Detections"):
        _FusionTracker().update(np.empty((0, 6)), detections_3d=spatial, camera=camera)


def test_sensor_identity_dimensions_and_mutable_storage_fail_before_kernel_entry() -> None:
    tracker = _FusionTracker()
    detections, spatial, camera = _inputs()
    with pytest.raises(ValueError, match="same sample"):
        tracker.update(detections, detections_3d=replace(spatial, sample_id="other"), camera=camera)
    frame = Frame(torch.zeros((3, 20, 30), dtype=torch.uint8), "camera:0")
    with pytest.raises(ValueError, match="spatial size"):
        tracker.update(detections, frame, detections_3d=spatial, camera=camera)
    spatial.geometry.values[0, 4] = -1.0
    with pytest.raises(ValueError, match="positive"):
        tracker.update(detections, detections_3d=spatial, camera=camera)
    assert tracker.frame_count == 0 and not tracker._first_frame_processed
    assert not hasattr(tracker, "received")


def test_both_sensor_class_catalogs_are_validated_without_advancing_state() -> None:
    detections, spatial, camera = _inputs()
    tracker = _FusionTracker(class_ids=(3, 2**40))
    with pytest.raises(ValueError, match="not present in the tracker class catalog"):
        tracker.update(detections, detections_3d=spatial, camera=camera)
    assert tracker.frame_count == 0 and not tracker._first_frame_processed


def test_multimodal_outputs_validate_independent_detection_indices() -> None:
    class _BadOutput(_FusionTracker):
        def _track_multimodal(self, *args):
            output = super()._track_multimodal(*args)
            return replace(output, spatial_tracks=replace(output.spatial_tracks, detection_indices=torch.tensor([1])))

    detections, spatial, camera = _inputs()
    with pytest.raises(ValueError, match="outside its current sensor batch"):
        _BadOutput().update(detections, detections_3d=spatial, camera=camera)
