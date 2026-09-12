"""Resolved multimodal tracker inputs follow the algorithm's enabled features."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from boxmot import EagerMot, MafHda
from boxmot.trackers.maf_hda import tracker as maf_hda
from tests.unit.trackers.multimodal.test_eagermot_package import _observations as _sensor_observations
from tests.unit.trackers.multimodal.test_maf_hda_package import _observations as _mask_observations


@pytest.mark.parametrize("per_class", [False, True])
def test_maf_motion_stages_track_masks_through_empty_frames_without_pixels(
    monkeypatch: pytest.MonkeyPatch, per_class: bool
) -> None:
    def forbid_appearance(*_args, **_kwargs):
        pytest.fail("Motion association must not construct appearance features.")

    monkeypatch.setattr(maf_hda, "MaskedKCF", forbid_appearance)
    tracker = MafHda(s2ta_mode="motion", t2ta_mode="motion", per_class=per_class)
    assert tracker.requirements.frame is False
    assert tracker.requirements.masks is True
    assert tracker.capabilities.requires_frame is False
    first, _frame = _mask_observations(0)
    output = tracker.update(first)
    assert len(output) == 2
    assert output.masks is not None
    second, _frame = _mask_observations(1)
    empty = tracker.update(second.select(torch.empty(0, dtype=torch.int64)))
    assert len(empty) == 0
    assert empty.masks.values.shape == (0, 64, 64)
    third, _frame = _mask_observations(2)
    resumed = tracker.update(third)
    assert len(resumed) == 2
    assert resumed.masks.values.shape == (2, 64, 64)
    assert resumed.track_ids.tolist() == output.track_ids.tolist()


@pytest.mark.parametrize("s2ta,t2ta", [("maf", "maf"), ("motion", "appearance"), ("appearance", "motion")])
def test_maf_enabled_appearance_requires_pixels_before_updating_state(s2ta: str, t2ta: str) -> None:
    tracker = MafHda(s2ta_mode=s2ta, t2ta_mode=t2ta)
    detections, _frame = _mask_observations(0)
    assert tracker.requirements.frame is True
    with pytest.raises(ValueError, match="requires a frame"):
        tracker.update(detections)
    assert tracker.frame_count == 0


def test_maf_motion_centroid_uses_frame_dimensions_without_decoding_pixels(monkeypatch: pytest.MonkeyPatch) -> None:
    tracker = MafHda(s2ta_mode="motion", t2ta_mode="motion", asso_func="centroid")
    assert tracker.requirements.frame is True
    assert tracker.requirements.frame_dimensions_only is True

    def forbid_pixels(*_args, **_kwargs):
        pytest.fail("Centroid association needs dimensions, not RGB pixels.")

    monkeypatch.setattr(tracker, "_frame_to_bgr", forbid_pixels)
    detections, frame = _mask_observations(0)
    result = tracker.update(detections, frame)
    assert len(result) == 2


def test_eager_ego_poses_are_optional_but_consumed_when_declared() -> None:
    detections, spatial, camera = _sensor_observations()
    empty_image = replace(detections.select(torch.empty(0, dtype=torch.int64)), masks=None)
    spatial = spatial.select(torch.tensor([1]))
    plain = EagerMot(distance_threshold=1.0)
    assert plain.requirements.camera is True
    assert plain.requirements.ego_motion is False
    assert plain.capabilities.accepts_ego_motion is True
    no_pose = plain.update(empty_image, detections_3d=spatial, camera=camera)
    assert len(no_pose.image_tracks) == 0 and len(no_pose.spatial_tracks) == 1
    torch.testing.assert_close(no_pose.spatial_tracks.geometry.values, spatial.geometry.values)

    tracker = EagerMot(distance_threshold=1.0)
    posed_camera = replace(camera, camera_to_world=torch.eye(4))
    first = tracker.update(empty_image, detections_3d=spatial, camera=posed_camera)
    moving_pose = torch.eye(4)
    moving_pose[0, 3] = 5
    spatial.geometry.values[:, 0] -= 5
    second = tracker.update(empty_image, detections_3d=spatial, camera=replace(camera, camera_to_world=moving_pose))
    assert second.spatial_tracks.track_ids.tolist() == first.spatial_tracks.track_ids.tolist()
    assert second.spatial_tracks.geometry.values[0, 0] == pytest.approx(-5)
    assert tracker._tracks[0].motion.box[0] == pytest.approx(0)

    with pytest.raises(ValueError, match="supplied consistently"):
        tracker.update(empty_image, detections_3d=spatial, camera=camera)
    assert tracker.frame_count == 2


@pytest.mark.parametrize("tracker_class", [EagerMot, MafHda])
def test_multimodal_trackers_explicitly_use_fixed_frame_timing(tracker_class: type) -> None:
    tracker = tracker_class()
    assert tracker.supports_variable_dt is False
    assert tracker.requirements.timestamp is False
    with pytest.raises(ValueError, match="does not support variable_dt"):
        tracker_class(variable_dt=True)


@pytest.mark.parametrize("tracker_class", [EagerMot, MafHda])
def test_multimodal_trackers_reject_unused_reid_embeddings_before_state_changes(tracker_class: type) -> None:
    tracker = tracker_class()
    if tracker_class is EagerMot:
        detections, spatial, camera = _sensor_observations()
        options = {"detections_3d": spatial, "camera": camera}
    else:
        detections, frame = _mask_observations(0)
        options = {"frame": frame}
    detections = replace(detections, embeddings=torch.ones((len(detections), 3), dtype=torch.float32))
    with pytest.raises(ValueError, match="does not use detection embeddings"):
        tracker.update(detections, **options)
    assert tracker.frame_count == 0
