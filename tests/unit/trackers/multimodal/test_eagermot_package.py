"""Public EagerMOT construction and independent sensor batch contracts."""

from __future__ import annotations

import importlib
from dataclasses import replace

import pytest
import torch

from boxmot import EagerMot
from boxmot.structures import Boxes, Boxes3D, CameraModel, Detections, Detections3D, MaskBatch, MultimodalTracks
from boxmot.trackers import TrackerRequirements, TrackerSpec, create_tracker
from boxmot.trackers.common.base import BaseTracker


def _observations(sample_id: str = "sequence:0") -> tuple[Detections, Detections3D, CameraModel]:
    """Place one image object beside an independent off-camera 3D observation."""
    masks = torch.zeros((1, 100, 200), dtype=torch.bool)
    masks[0, 39:62, 78:122] = True
    image_detections = Detections(
        geometry=Boxes(torch.tensor([[78, 39, 122, 62]], dtype=torch.float32)),
        scores=torch.tensor([0.95]),
        class_ids=torch.tensor([5]),
        sample_id=sample_id,
        masks=MaskBatch(masks),
    )
    spatial_detections = Detections3D(
        geometry=Boxes3D(torch.tensor([[40, 1, 10, 0, 4, 2, 2], [0, 1, 10, 0, 4, 2, 2]], dtype=torch.float32)),
        scores=torch.tensor([0.9, 0.9]),
        class_ids=torch.tensor([7, 5]),
        sample_id=sample_id,
    )
    camera = CameraModel(
        projection=torch.tensor([[100, 0, 100, 0], [0, 100, 50, 0], [0, 0, 1, 0]], dtype=torch.float32),
        image_size=(100, 200),
    )
    return image_detections, spatial_detections, camera


def test_eagermot_uses_the_shared_validated_update_boundary() -> None:
    """Public imports and the factory expose the registered domain class."""
    tracker = create_tracker(TrackerSpec("eagermot"))
    implementation = importlib.import_module("boxmot.trackers.eagermot.tracker")
    package = importlib.import_module("boxmot.trackers.eagermot")

    assert type(tracker) is EagerMot is implementation.EagerMot
    assert EagerMot.update is BaseTracker.update
    assert not hasattr(package, "EagerMot")
    assert tracker.requirements == TrackerRequirements(detections_3d=True, camera=True)


@pytest.mark.parametrize("per_class", (False, True))
def test_eagermot_keeps_sensor_indices_and_masks_independently_aligned(per_class: bool) -> None:
    """Off-camera spatial output needs no fabricated image box or mask."""
    tracker = create_tracker(TrackerSpec("eagermot", per_class=per_class))
    detections, spatial, camera = _observations()
    result = tracker.update(detections, detections_3d=spatial, camera=camera)

    assert isinstance(result, MultimodalTracks)
    assert len(result.image_tracks) == 1
    assert len(result.spatial_tracks) == 2
    assert result.image_tracks.detection_indices.tolist() == [0]
    assert sorted(result.spatial_tracks.detection_indices.tolist()) == [0, 1]
    image_id = result.image_tracks.track_ids.item()
    spatial_index = result.spatial_tracks.track_ids.tolist().index(image_id)
    assert result.spatial_tracks.class_ids[spatial_index] == 5
    assert result.spatial_tracks.detection_indices[spatial_index] == 1
    torch.testing.assert_close(result.image_tracks.masks.values, detections.masks.values)

    tracker.reset()
    restarted = tracker.update(detections, detections_3d=spatial, camera=camera)
    torch.testing.assert_close(restarted.spatial_tracks.track_ids, result.spatial_tracks.track_ids)


def test_eagermot_requires_explicit_spatial_context_before_mutating_state() -> None:
    """A missing detector batch differs from a completed empty detector batch."""
    tracker = EagerMot()
    detections, spatial, camera = _observations()
    with pytest.raises(ValueError, match="requires detections_3d"):
        tracker.update(detections, camera=camera)
    with pytest.raises(ValueError, match="requires camera"):
        tracker.update(detections, detections_3d=spatial)
    assert tracker.frame_count == 0

    empty = spatial.select(torch.empty(0, dtype=torch.int64))
    result = tracker.update(detections, detections_3d=empty, camera=camera)
    assert len(result) == 0
    assert tracker.frame_count == 1


def test_eagermot_accepts_sensor_dropout_without_requiring_pixels_or_masks() -> None:
    """A 2D observation keeps its shared identity through one 3D dropout."""
    tracker = EagerMot()
    detections, spatial, camera = _observations()
    detections = replace(detections, masks=None)
    first = tracker.update(detections, detections_3d=spatial, camera=camera)
    detections = replace(detections, sample_id="sequence:1")
    empty = replace(spatial.select(torch.empty(0, dtype=torch.int64)), sample_id=detections.sample_id)
    second = tracker.update(detections, detections_3d=empty, camera=camera)

    assert second.image_tracks.track_ids.tolist() == first.image_tracks.track_ids.tolist()
    assert second.spatial_tracks.track_ids.tolist() == second.image_tracks.track_ids.tolist()
    assert second.spatial_tracks.detection_indices.tolist() == [-1]
    assert second.image_tracks.masks is None


def test_eagermot_rejects_unsupported_runtime_choices() -> None:
    """Unsupported geometry, native execution, and image metrics fail explicitly."""
    with pytest.raises(ValueError, match="does not support geometry kind 'obb'"):
        create_tracker(TrackerSpec("eagermot", geometry="obb"))
    with pytest.raises(ValueError, match="does not support OBB geometry"):
        EagerMot(is_obb=True)
    with pytest.raises(ValueError, match="Native backend is unavailable"):
        create_tracker(TrackerSpec("eagermot", backend="cpp"))
    with pytest.raises(ValueError, match="only asso_func='iou'"):
        create_tracker(TrackerSpec("eagermot", options=(("asso_func", "giou"),)))
