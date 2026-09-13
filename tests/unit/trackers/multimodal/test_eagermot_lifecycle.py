"""Sensor fusion, dropout recovery, provenance, and lifecycle behavior."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from boxmot import EagerMot
from boxmot.structures import Boxes, Boxes3D, CameraModel, Detections, Detections3D, MaskBatch, MultimodalTracks


def _camera(translation: float | None = None) -> CameraModel:
    """A small pinhole camera, optionally translated along the world x axis."""
    pose = None
    if translation is not None:
        pose = torch.eye(4, dtype=torch.float32)
        pose[0, 3] = translation
    return CameraModel(
        projection=torch.tensor([[100, 0, 64, 0], [0, 100, 48, 0], [0, 0, 1, 0]], dtype=torch.float32),
        image_size=(96, 128),
        camera_to_world=pose,
    )


def _detections(
    frame: int,
    boxes: list[list[float]],
    *,
    scores: list[float] | None = None,
    classes: list[int] | None = None,
    masks: bool = False,
) -> Detections:
    """Build independent image detections with optional rectangular masks."""
    values = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    mask_values = None
    if masks:
        mask_values = torch.zeros((len(boxes), 96, 128), dtype=torch.bool)
        for index, (x1, y1, x2, y2) in enumerate(values.to(torch.int64).tolist()):
            mask_values[index, y1:y2, x1:x2] = True
    return Detections(
        geometry=Boxes(values),
        scores=torch.tensor([0.9] * len(boxes) if scores is None else scores, dtype=torch.float32),
        class_ids=torch.tensor([0] * len(boxes) if classes is None else classes, dtype=torch.int64),
        sample_id=f"test/{frame}",
        masks=None if mask_values is None else MaskBatch(mask_values),
    )


def _spatial(
    frame: int,
    xs: list[float],
    *,
    scores: list[float] | None = None,
    classes: list[int] | None = None,
    depth: float = 20,
) -> Detections3D:
    """Build independently counted upright car boxes in camera coordinates."""
    boxes = torch.tensor([[x, 1, depth, 0, 4, 2, 2] for x in xs], dtype=torch.float32).reshape(-1, 7)
    return Detections3D(
        geometry=Boxes3D(boxes),
        scores=torch.tensor([0.95] * len(xs) if scores is None else scores, dtype=torch.float32),
        class_ids=torch.tensor([0] * len(xs) if classes is None else classes, dtype=torch.int64),
        sample_id=f"test/{frame}",
    )


@pytest.mark.parametrize("per_class", (False, True))
def test_fused_tracks_preserve_masks_and_independent_sensor_indices(per_class: bool) -> None:
    tracker = EagerMot(det_thresh=0.5, det_thresh_3d=0.5, per_class=per_class)
    image = _detections(0, [[0, 0, 10, 10], [53, 43, 75, 53]], scores=[0.1, 0.8], masks=True)
    spatial = _spatial(0, [50, 0], scores=[0.1, 0.95])
    original_geometry, original_masks = spatial.geometry.values.clone(), image.masks.values.clone()
    result = tracker.update(image, detections_3d=spatial, camera=_camera())
    assert isinstance(result, MultimodalTracks)
    assert result.image_tracks.track_ids.tolist() == result.spatial_tracks.track_ids.tolist() == [0]
    assert result.image_tracks.detection_indices.tolist() == result.spatial_tracks.detection_indices.tolist() == [1]
    assert result.image_tracks.scores.item() == pytest.approx(0.8)
    assert result.spatial_tracks.scores.item() == pytest.approx(0.95)
    torch.testing.assert_close(result.image_tracks.masks.values, original_masks[1:])
    torch.testing.assert_close(spatial.geometry.values, original_geometry)
    torch.testing.assert_close(image.masks.values, original_masks)
    assert tracker.get_active_tracks_for_display()[0].id == 0


def test_2d_only_observations_cannot_initialize_tracks() -> None:
    tracker = EagerMot()
    result = tracker.update(
        _detections(0, [[53, 43, 75, 53]], masks=True), detections_3d=_spatial(0, []), camera=_camera()
    )
    assert len(result.image_tracks) == len(result.spatial_tracks) == 0
    assert result.image_tracks.masks.values.shape == (0, 96, 128)
    assert tracker.id_allocator.next_id == 0


@pytest.mark.parametrize("depth", (20, -20))
def test_3d_only_tracks_do_not_require_a_visible_image_box(depth: float) -> None:
    tracker = EagerMot()
    result = tracker.update(_detections(0, []), detections_3d=_spatial(0, [0], depth=depth), camera=_camera())
    assert len(result.image_tracks) == 0
    assert result.spatial_tracks.track_ids.tolist() == [0]
    assert result.spatial_tracks.geometry.values[0, 2].item() == depth
    assert result.spatial_tracks.detection_indices.tolist() == [0]
    assert result.spatial_tracks.scores.item() == pytest.approx(0.95 / 16)


@pytest.mark.parametrize("first_matching_method", ("dist_2d", "dist_2d_dims", "dist_2d_full", "iou_3d"))
def test_2d_recovery_retains_identity_and_predicts_through_depth_dropout(first_matching_method: str) -> None:
    tracker = EagerMot(first_matching_method=first_matching_method)
    first = tracker.update(
        _detections(0, [[53, 43, 75, 53]], masks=True), detections_3d=_spatial(0, [0]), camera=_camera()
    )
    tracker.update(_detections(1, [[59, 43, 80, 53]]), detections_3d=_spatial(1, [1]), camera=_camera())
    before = tracker._tracks[0].motion.box.copy()
    recovered = tracker.update(
        _detections(2, [[64, 43, 85, 53]], masks=True), detections_3d=_spatial(2, []), camera=_camera()
    )
    assert recovered.image_tracks.track_ids.tolist() == first.image_tracks.track_ids.tolist()
    assert recovered.spatial_tracks.track_ids.tolist() == first.spatial_tracks.track_ids.tolist()
    assert recovered.image_tracks.detection_indices.tolist() == [0]
    assert recovered.spatial_tracks.detection_indices.tolist() == [-1]
    assert recovered.spatial_tracks.geometry.values[0, 0].item() > before[0] + 0.5
    assert tracker._tracks[0].hits == 3
    assert recovered.spatial_tracks.scores.item() == pytest.approx(0.95)


def test_motion_advances_on_completely_empty_frames_without_emitting_predictions() -> None:
    tracker = EagerMot(max_age=4)
    tracker.update(_detections(0, []), detections_3d=_spatial(0, [0]), camera=_camera())
    tracker.update(_detections(1, []), detections_3d=_spatial(1, [1]), camera=_camera())
    before = tracker._tracks[0].motion.box[0]
    for frame in (2, 3):
        result = tracker.update(_detections(frame, []), detections_3d=_spatial(frame, []), camera=_camera())
        assert len(result.image_tracks) == len(result.spatial_tracks) == 0
    assert tracker._tracks[0].motion.box[0] > before + 1.5
    result = tracker.update(_detections(4, []), detections_3d=_spatial(4, [4]), camera=_camera())
    assert result.spatial_tracks.track_ids.tolist() == [0]


@pytest.mark.parametrize("missing_frames", (1, 2, 3))
def test_expiry_occurs_only_after_max_age_frames_without_either_sensor(missing_frames: int) -> None:
    tracker = EagerMot(max_age=3)
    tracker.update(_detections(0, []), detections_3d=_spatial(0, [0]), camera=_camera())
    for frame in range(1, missing_frames + 1):
        tracker.update(_detections(frame, []), detections_3d=_spatial(frame, []), camera=_camera())
    frame = missing_frames + 1
    result = tracker.update(_detections(frame, []), detections_3d=_spatial(frame, [0]), camera=_camera())
    assert result.spatial_tracks.track_ids.tolist() == ([0] if missing_frames < 3 else [1])


@pytest.mark.parametrize("per_class", (False, True))
def test_fusion_and_association_preserve_exact_class_ids(per_class: bool) -> None:
    classes = [2**55 + 1, 7]
    tracker = EagerMot(per_class=per_class, class_ids=classes)
    for frame in range(3):
        image = _detections(frame, [[53, 43, 75, 53]] * 2, classes=classes[::-1])
        spatial = _spatial(frame, [0, 0], classes=classes)
        result = tracker.update(image, detections_3d=spatial, camera=_camera())
        assert result.spatial_tracks.track_ids.tolist() == [0, 1]
        assert result.spatial_tracks.class_ids.tolist() == classes
        assert result.image_tracks.class_ids.tolist() == classes
        assert result.image_tracks.detection_indices.tolist() == [1, 0]


def test_2d_recovery_is_class_gated() -> None:
    tracker = EagerMot()
    tracker.update(_detections(0, []), detections_3d=_spatial(0, [0], classes=[1]), camera=_camera())
    result = tracker.update(
        _detections(1, [[53, 43, 75, 53]], classes=[2]), detections_3d=_spatial(1, []), camera=_camera()
    )
    assert len(result.spatial_tracks) == 0
    assert tracker._tracks[0].time_since_update == 1


def test_per_class_queries_preserve_canonical_ids_and_follow_lifecycle_and_reset() -> None:
    classes = [2**55 + 1, 7]
    tracker = EagerMot(per_class=True, class_ids=classes)
    tracker.update(
        _detections(0, [[53, 43, 75, 53]] * 2, classes=classes),
        detections_3d=_spatial(0, [0, 0], classes=classes),
        camera=_camera(),
    )
    for track_id, class_id in enumerate(classes):
        assert [track.id for track in tracker.get_class_tracks(class_id)] == [track_id]
        assert [track.cls for track in tracker.get_class_tracks(class_id, "pool")] == [class_id]
    assert {track.id for track in tracker.all_class_tracks()} == {0, 1}
    assert {track.id for track in tracker.all_class_tracks("pool")} == {0, 1}

    tracker.update(_detections(1, []), detections_3d=_spatial(1, []), camera=_camera())
    assert tracker.all_class_tracks() == []
    assert {track.id for track in tracker.all_class_tracks("pool")} == {0, 1}
    for class_id in classes:
        assert tracker.get_class_tracks(class_id) == []
        assert len(tracker.get_class_tracks(class_id, "pool")) == 1

    tracker.reset()
    assert tracker.all_class_tracks("pool") == []
    assert tracker.get_class_tracks(classes[0], "pool") == []
    tracker.update(_detections(2, []), detections_3d=_spatial(2, [0], classes=[classes[1]]), camera=_camera())
    assert [track.id for track in tracker.get_class_tracks(classes[1], "pool")] == [0]
    assert tracker.get_class_tracks(classes[0], "pool") == []


def test_second_stage_can_be_disabled_using_the_source_threshold_setting() -> None:
    tracker = EagerMot(iou_threshold=1.0)
    tracker.update(_detections(0, [[53, 43, 75, 53]]), detections_3d=_spatial(0, [0]), camera=_camera())
    result = tracker.update(_detections(1, [[53, 43, 75, 53]]), detections_3d=_spatial(1, []), camera=_camera())
    assert len(result.image_tracks) == len(result.spatial_tracks) == 0


def test_unmatched_fused_3d_observation_cannot_bypass_the_3d_gate() -> None:
    tracker = EagerMot(distance_threshold=0.1)
    tracker.update(_detections(0, [[53, 43, 75, 53]]), detections_3d=_spatial(0, [0]), camera=_camera())
    result = tracker.update(_detections(1, [[56, 43, 78, 53]]), detections_3d=_spatial(1, [0.5]), camera=_camera())
    assert result.spatial_tracks.track_ids.tolist() == [1]
    assert result.image_tracks.track_ids.tolist() == [1]
    assert tracker._tracks[0].time_since_update == 1


def test_image_support_restores_confidence_after_3d_only_updates() -> None:
    tracker = EagerMot(max_age_2d=2)
    tracker.update(_detections(0, [[53, 43, 75, 53]]), detections_3d=_spatial(0, [0]), camera=_camera())
    for frame, expected in ((1, 0.95), (2, 0.95 / 2), (3, 0.95 / 4)):
        result = tracker.update(_detections(frame, []), detections_3d=_spatial(frame, [0]), camera=_camera())
        assert result.spatial_tracks.scores.item() == pytest.approx(expected)
    restored = tracker.update(_detections(4, [[53, 43, 75, 53]]), detections_3d=_spatial(4, []), camera=_camera())
    assert restored.spatial_tracks.scores.item() == pytest.approx(0.95)


def test_confirmation_counts_observations_after_initial_warmup() -> None:
    tracker = EagerMot(min_hits=3)
    for frame in range(3):
        tracker.update(_detections(frame, []), detections_3d=_spatial(frame, []), camera=_camera())
    for frame in (3, 4, 5):
        result = tracker.update(_detections(frame, []), detections_3d=_spatial(frame, [0]), camera=_camera())
        assert len(result.spatial_tracks) == (1 if frame == 5 else 0)


def test_initial_warmup_matches_released_source_confirmation() -> None:
    tracker = EagerMot(min_hits=3)
    result = tracker.update(_detections(0, []), detections_3d=_spatial(0, [0]), camera=_camera())
    assert result.spatial_tracks.track_ids.tolist() == [0]
    tracker.update(_detections(1, []), detections_3d=_spatial(1, []), camera=_camera())
    result = tracker.update(_detections(2, []), detections_3d=_spatial(2, [0]), camera=_camera())
    assert len(result.spatial_tracks) == 0


def test_world_motion_compensates_ego_translation_and_returns_camera_coordinates() -> None:
    tracker = EagerMot(distance_threshold=0.1)
    for frame, translation in enumerate((0.0, 3.0, 6.0)):
        spatial = _spatial(frame, [5 - translation])
        original = spatial.geometry.values.clone()
        result = tracker.update(_detections(frame, []), detections_3d=spatial, camera=_camera(translation))
        assert result.spatial_tracks.track_ids.tolist() == [0]
        assert result.spatial_tracks.geometry.values[0, 0].item() == pytest.approx(5 - translation)
        assert tracker._tracks[0].motion.box[0] == pytest.approx(5)
        torch.testing.assert_close(spatial.geometry.values, original)


def test_changing_pose_mode_is_rejected_before_advancing_tracker() -> None:
    tracker = EagerMot()
    tracker.update(_detections(0, []), detections_3d=_spatial(0, [0]), camera=_camera())
    with pytest.raises(ValueError, match="consistently"):
        tracker.update(_detections(1, []), detections_3d=_spatial(1, [0]), camera=_camera(0.0))
    assert tracker.frame_count == 1
    assert tracker._tracks[0].hits == 1
    tracker.reset()
    result = tracker.update(_detections(1, []), detections_3d=_spatial(1, [0]), camera=_camera(0.0))
    assert result.spatial_tracks.track_ids.tolist() == [0]


def test_reset_and_multiple_instances_have_independent_ids() -> None:
    first, second = EagerMot(), EagerMot()
    for tracker in (first, second):
        result = tracker.update(_detections(0, []), detections_3d=_spatial(0, [0]), camera=_camera())
        assert result.spatial_tracks.track_ids.tolist() == [0]
    first.reset()
    result = first.update(_detections(1, []), detections_3d=_spatial(1, [0]), camera=_camera())
    assert result.spatial_tracks.track_ids.tolist() == [0]
    assert len(second._tracks) == 1


@pytest.mark.parametrize(
    "options",
    (
        {"det_thresh": -0.1},
        {"det_thresh_3d": np.nan},
        {"max_age": 0},
        {"max_age_2d": True},
        {"min_hits": 1.5},
        {"distance_threshold": 0},
        {"first_matching_method": "unknown"},
        {"is_angular": 1},
        {"asso_func": "giou"},
        {"is_obb": True},
        {"variable_dt": True},
    ),
)
def test_unsupported_or_invalid_configuration_is_rejected(options: dict) -> None:
    with pytest.raises((TypeError, ValueError)):
        EagerMot(**options)
