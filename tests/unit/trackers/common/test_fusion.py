"""Association fusion preserves geometry, statistical gates, and forbidden pairs."""

from types import SimpleNamespace

import numpy as np
import pytest

from boxmot.trackers.common.association import (
    apply_motion_fusion,
    fuse_iou,
    fuse_motion,
    fuse_score,
    gate_cost_matrix,
    motion_gating_distance,
    motion_gating_threshold,
)
from boxmot.trackers.common.motion.kalman_filters.noise import KalmanNoiseConfig
from boxmot.trackers.common.motion.models import MotionModelKind, create_motion_model


def _track(kind: MotionModelKind, is_obb: bool, *, noise_scale: float = 1.0) -> tuple[SimpleNamespace, np.ndarray]:
    """Build an initialized distribution using the same adapter as live tracks."""
    box = np.array([20.0, 30.0, 40.0, 70.0])
    if is_obb:
        box = np.array([30.0, 50.0, 20.0, 40.0, np.pi - 0.01])
    model = create_motion_model(kind, is_obb=is_obb)
    kalman = model.create_filter(noise_config=KalmanNoiseConfig(measurement_noise_scale=noise_scale))
    measurement = model.to_measurement(np.r_[box, 0.9], column=False)
    mean, covariance = model.initiate(kalman, measurement, confidence=0.9)
    return SimpleNamespace(motion_model=model, kalman_filter=kalman, mean=mean, covariance=covariance), box


@pytest.mark.parametrize("kind", tuple(MotionModelKind))
@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("ownership", ["box", "strongsort", "matrix"])
def test_motion_gate_uses_selected_model_projection_and_track_distribution(kind, is_obb, ownership):
    track, box = _track(kind, is_obb)
    mean, covariance = track.mean.copy(), track.covariance.copy()
    projected_covariance = track.motion_model.project(track.kalman_filter, mean, covariance)[1]
    if ownership == "strongsort":
        track.kf = track.kalman_filter
        del track.kalman_filter
    elif ownership == "matrix":
        track.kf = track.kalman_filter
        track.kf.x, track.kf.P = mean[:, None], covariance
        del track.kalman_filter, track.mean, track.covariance

    boxes = np.repeat(box[None], 2, axis=0)
    boxes[1, 0] += 2.0
    if not is_obb:
        boxes[1, 2] += 2.0
    original_boxes = boxes.copy()
    distance = motion_gating_distance(track, boxes)
    expected = 4.0 / projected_covariance[0, 0]
    np.testing.assert_allclose(distance, [0.0, expected], atol=1e-12)
    np.testing.assert_array_equal(boxes, original_boxes)
    np.testing.assert_array_equal(getattr(track, "mean", mean), mean)
    np.testing.assert_array_equal(getattr(track, "covariance", covariance), covariance)
    assert motion_gating_threshold(track) == (11.070 if is_obb else 9.4877)
    assert motion_gating_threshold(track, only_position=True) == 5.9915


@pytest.mark.parametrize("kind", tuple(MotionModelKind))
@pytest.mark.parametrize("is_obb", [False, True])
def test_motion_gate_uses_track_specific_measurement_noise(kind, is_obb):
    narrow, box = _track(kind, is_obb)
    broad, _ = _track(kind, is_obb, noise_scale=20.0)
    box[0] += 15.0
    if not is_obb:
        box[2] += 15.0
    narrow_distance = motion_gating_distance(narrow, box[None])[0]
    broad_distance = motion_gating_distance(broad, box[None])[0]
    assert 0.0 < broad_distance < narrow_distance


@pytest.mark.parametrize("kind", tuple(MotionModelKind))
def test_motion_gate_aligns_equivalent_obb_forms_and_angle_wrap(kind):
    track, box = _track(kind, True)
    equivalents = np.repeat(box[None], 4, axis=0)
    equivalents[1, 4] += np.pi
    equivalents[2:, 2:4] = box[[3, 2]]
    equivalents[2, 4] += np.pi / 2
    equivalents[3, 4] -= np.pi / 2
    original_boxes = equivalents.copy()
    np.testing.assert_allclose(motion_gating_distance(track, equivalents), 0.0, atol=1e-12)
    np.testing.assert_array_equal(equivalents, original_boxes)

    wrapped = box.copy()
    wrapped[4] = -np.pi + 0.01
    continuous = wrapped.copy()
    continuous[4] += 2 * np.pi
    distances = motion_gating_distance(track, np.stack((wrapped, continuous)))
    assert 0.0 < distances[0] < motion_gating_threshold(track)
    np.testing.assert_allclose(distances[0], distances[1], atol=1e-12)


@pytest.mark.parametrize("kind", tuple(MotionModelKind))
@pytest.mark.parametrize("is_obb", [False, True])
def test_position_gate_excludes_shape(kind, is_obb):
    track, box = _track(kind, is_obb)
    if is_obb:
        box[2:4] *= 10
    else:
        box[:2] -= 100
        box[2:] += 100
    np.testing.assert_allclose(motion_gating_distance(track, box[None], only_position=True), 0.0)
    assert motion_gating_distance(track, box[None])[0] > motion_gating_threshold(track)


@pytest.mark.parametrize("is_obb", [False, True])
def test_xyscr_gate_marginalizes_confidence_instead_of_conditioning_on_it(is_obb):
    track, box = _track(MotionModelKind.XYSCR, is_obb)
    boxes = box[None].copy()
    boxes[0, 0] += 3.0
    if not is_obb:
        boxes[0, 2] += 3.0
    original_distance = motion_gating_distance(track, boxes)
    track.mean[3] = -1000.0
    # Confidence/position correlation must not tighten the geometric gate.
    track.covariance[0, 3] = track.covariance[3, 0] = 5.0
    np.testing.assert_allclose(motion_gating_distance(track, boxes), original_distance)


@pytest.mark.parametrize("lambda_", [0.0, 0.25, 0.98, 1.0])
def test_motion_fusion_applies_gate_and_raw_mahalanobis_blend_without_mutation(lambda_):
    track, box = _track(MotionModelKind.XYWH, False)
    boxes = np.repeat(box[None], 4, axis=0)
    boxes[1, [0, 2]] += 1.0
    boxes[2, [0, 2]] += 1000.0
    costs = np.array([[0.2, 0.4, 0.0, np.inf]])
    original_costs = costs.copy()
    distances = motion_gating_distance(track, boxes)
    with np.errstate(invalid="raise"):
        fused = fuse_motion(costs, [track], boxes, lambda_=lambda_)
    np.testing.assert_allclose(fused[0, :2], lambda_ * costs[0, :2] + (1 - lambda_) * distances[:2])
    assert np.isinf(fused[0, 2:]).all()
    np.testing.assert_array_equal(costs, original_costs)
    np.testing.assert_array_equal(gate_cost_matrix(costs, [track], boxes), [[0.2, 0.4, np.inf, np.inf]])


def test_gate_boundary_is_inclusive(monkeypatch):
    track, box = _track(MotionModelKind.XYAH, False)
    threshold = motion_gating_threshold(track)
    monkeypatch.setattr(
        "boxmot.trackers.common.association.matching.motion_gating_distance",
        lambda *args, **kwargs: np.array([threshold, np.nextafter(threshold, np.inf)]),
    )
    result = gate_cost_matrix(np.array([[0.2, 0.2]]), [track], np.repeat(box[None], 2, axis=0))
    np.testing.assert_array_equal(result, [[0.2, np.inf]])


@pytest.mark.parametrize("lambda_", [-0.1, 1.1, np.nan, np.inf])
def test_motion_fusion_rejects_invalid_blend_weight(lambda_):
    with pytest.raises(ValueError, match="lambda_"):
        fuse_motion(np.empty((0, 0)), [], np.empty((0, 4)), lambda_=lambda_)


@pytest.mark.parametrize("shape", [(0, 0), (0, 3), (2, 0)])
def test_fusions_preserve_empty_matrix_shape(shape):
    costs = np.empty(shape)
    tracks = [object()] * shape[0]
    boxes = np.empty((shape[1], 4))
    for result in (
        fuse_score(costs, np.zeros(shape[1])),
        fuse_iou(costs, tracks, boxes),
        fuse_motion(costs, tracks, boxes),
        gate_cost_matrix(costs, tracks, boxes),
    ):
        assert result.shape == shape


def test_iou_fusion_combines_appearance_with_geometry_and_preserves_forbidden_edges():
    tracks = np.array([[0.0, 0.0, 10.0, 10.0]])
    detections = np.array([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0], [0.0, 0.0, 10.0, 10.0]])
    costs = np.array([[0.2, 0.4, np.inf]])
    np.testing.assert_allclose(fuse_iou(costs, tracks, detections), [[0.2, 0.7, np.inf]])
    np.testing.assert_array_equal(costs, [[0.2, 0.4, np.inf]])


def test_iou_fusion_uses_oriented_geometry():
    tracks = np.array([[10.0, 10.0, 20.0, 2.0, 0.0]])
    detections = tracks.copy()
    detections[:, 4] = np.pi / 2
    result = fuse_iou(np.zeros((1, 1)), tracks, detections, is_obb=True)
    assert result[0, 0] > 0.4


@pytest.mark.parametrize("track_objects", [False, True])
@pytest.mark.parametrize("detection_objects", [False, True])
def test_iou_fusion_accepts_independent_object_and_box_collections(track_objects, detection_objects):
    track_boxes = np.array([[0.0, 0.0, 10.0, 10.0]])
    detection_boxes = np.array([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    tracks = [SimpleNamespace(xyxy=box, is_obb=False) for box in track_boxes] if track_objects else track_boxes
    detections = (
        tuple(SimpleNamespace(xyxy=box, is_obb=False) for box in detection_boxes)
        if detection_objects else detection_boxes
    )
    costs = np.array([[0.2, 0.4]])
    np.testing.assert_allclose(fuse_iou(costs, tracks, detections), [[0.2, 0.7]])
    np.testing.assert_array_equal(track_boxes, [[0.0, 0.0, 10.0, 10.0]])
    np.testing.assert_array_equal(detection_boxes, [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])


@pytest.mark.parametrize("object_side", ["tracks", "detections"])
def test_iou_fusion_infers_obb_from_either_object_collection(object_side):
    track_boxes = np.array([[10.0, 10.0, 20.0, 2.0, 0.0]])
    detection_boxes = np.array([[10.0, 10.0, 20.0, 2.0, np.pi / 2]])
    tracks = (
        tuple(SimpleNamespace(xywha=box, is_obb=True) for box in track_boxes)
        if object_side == "tracks" else track_boxes
    )
    detections = (
        [SimpleNamespace(xywha=box, is_obb=True) for box in detection_boxes]
        if object_side == "detections" else detection_boxes
    )
    costs = np.zeros((1, 1))
    result = fuse_iou(costs, tracks, detections)
    np.testing.assert_allclose(result, fuse_iou(costs, track_boxes, detection_boxes, is_obb=True))
    assert result[0, 0] > 0.4


@pytest.mark.parametrize("objects", [False, True])
def test_score_fusion_accepts_confidence_arrays_and_objects_without_nan_for_forbidden_zero_score(objects):
    confidence = np.array([0.8, 0.0, 0.0])
    detections = [SimpleNamespace(conf=value) for value in confidence] if objects else confidence
    costs = np.array([[0.2, 0.4, np.inf]])
    with np.errstate(invalid="raise"):
        result = fuse_score(costs, detections)
    np.testing.assert_allclose(result, [[0.36, 1.0, np.inf]])
    np.testing.assert_array_equal(costs, [[0.2, 0.4, np.inf]])


def test_motion_policy_off_does_not_require_kalman_tracks():
    costs = np.zeros((1, 1))
    assert apply_motion_fusion(costs, [object()], np.zeros((1, 4)), mode="off") is costs
    with pytest.raises(ValueError, match="Unknown motion fusion mode"):
        apply_motion_fusion(costs, [object()], np.zeros((1, 4)), mode="unknown")
