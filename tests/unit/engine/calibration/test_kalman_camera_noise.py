"""Camera-aware covariance fits separate object diffusion from image motion."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pytest

from boxmot.engine.calibration.kalman import calibrate_kalman
from boxmot.engine.calibration.kalman_compensation import validate_camera_warps
from boxmot.engine.calibration.kalman_data import CalibrationTrack
from boxmot.engine.calibration.kalman_fit import _collect_track_moments, fit_kalman_noise
from boxmot.engine.calibration.kalman_model import CalibrationModel
from boxmot.trackers.common.motion.kalman_filters.fitting import CoupledProcessNoiseMoments, ScalarNoiseMoments
from boxmot.trackers.common.motion.models import supported_motion_model_kinds
from tests.unit.engine.calibration.test_kalman import _args, _data


def _track(
    boxes: np.ndarray, *, indices: np.ndarray | None = None, times: np.ndarray | None = None
) -> CalibrationTrack:
    return CalibrationTrack(
        sequence_id="camera",
        track_id=1,
        class_id=0,
        frame_indices=np.arange(len(boxes)) if indices is None else indices,
        timestamps_s=times,
        gt_boxes=boxes,
        detection_boxes=boxes.copy(),
        scores=np.full(len(boxes), 0.8),
    )


def _warps(count: int) -> np.ndarray:
    return np.tile(np.eye(3)[:2], (count, 1, 1))


@pytest.mark.parametrize(
    "geometry,kind",
    [
        (geometry, kind.value)
        for geometry in ("aabb", "obb")
        for kind in supported_motion_model_kinds(is_obb=geometry == "obb")
    ],
)
def test_runtime_camera_only_motion_needs_no_object_process_or_birth_velocity(kind: str, geometry: str) -> None:
    model = CalibrationModel("bytetrack", geometry, {"kalman.type": kind})
    box = np.array([150.0, 120.0, 32.0, 64.0, 3.1] if geometry == "obb" else [130.0, 88.0, 162.0, 152.0])
    state, _ = model.initial_state(model.to_measurement(box))
    warps = _warps(30)
    boxes = [box]
    for index in range(1, len(warps)):
        angle, scale = 0.015 * np.sin(index), 1.01 if index % 2 else 1.0 / 1.01
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        warps[index, :, :2] = scale * rotation
        warps[index, :, 2] = [2.0 * np.sin(index / 3), -3.0 * np.cos(index / 3)]
        state, _ = model.camera_transform(state, warps[index])
        boxes.append(model.motion_model.to_box(state)[0])
    track = _track(np.asarray(boxes))
    camera, statistics = fit_kalman_noise([track, track], {0: model}, model.options, camera_warps={"camera": warps})
    raw, _ = fit_kalman_noise([track, track], {0: model}, model.options)
    assert statistics == {"gt_transitions": 2 * (len(boxes) - 2), "gt_lag_pairs": 2 * (len(boxes) - 3)}
    for option in ("process_position_diagonal", "process_velocity_diagonal", "initial_velocity_diagonal"):
        estimates = camera[f"kalman.noise.{option}"]["coordinates"]
        assert all(item["status"] in {"fitted", "excluded"} for item in estimates)
        assert all(item["value"] < 1e-5 for item in estimates if item["status"] != "excluded")
    for option in ("measurement_noise_diagonal", "initial_position_diagonal"):
        assert camera[f"kalman.noise.{option}"] == raw[f"kalman.noise.{option}"]
    assert max(raw["kalman.noise.process_position_diagonal"]["value"][:2]) > 1e-4


@pytest.mark.parametrize("kind", [kind.value for kind in supported_motion_model_kinds(is_obb=True)])
@pytest.mark.parametrize("moving", [False, True])
def test_large_camera_rotations_preserve_obb_process_and_birth_velocity(kind: str, moving: bool) -> None:
    """Equivalent rectangles retain the camera's branch, including moving objects."""
    model = CalibrationModel("bytetrack", "obb", {"kalman.type": kind})
    box = np.array([150.0, 120.0, 32.0, 64.0, 0.2])
    state, initial_covariance = model.initial_state(model.to_measurement(box))
    velocity = np.asarray(model.velocity_indices)
    if moving:
        state[velocity] = 0.01
        state[velocity[:2]] = [0.5, -0.25]
    birth_velocity = state[velocity].copy()
    warps = _warps(30)
    boxes = [box]
    for index in range(1, len(warps)):
        angle = 1.1 + 0.1 * np.sin(index)
        scale = 1.01 if index % 2 else 1.0 / 1.01
        warps[index, :, :2] = scale * np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        warps[index, :, 2] = [2.0, -3.0]
        state, _ = model.camera_transform(model.transition() @ state, warps[index])
        boxes.append(model.motion_model.to_box(state)[0])
    track = _track(np.asarray(boxes))
    # GT and detector encodings independently choose equivalent rectangle forms.
    track.gt_boxes[1::2, 2:4] = track.gt_boxes[1::2, 3:1:-1]
    track.gt_boxes[1::2, 4] += np.pi / 2.0
    track.detection_boxes[::2, 2:4] = track.detection_boxes[::2, 3:1:-1]
    track.detection_boxes[::2, 4] -= np.pi / 2.0
    track.detection_boxes[1::2, 4] += np.pi
    fitted, _ = fit_kalman_noise([track, track], {0: model}, model.options, camera_warps={"camera": warps})
    for option in ("process_position_diagonal", "process_velocity_diagonal", "measurement_noise_diagonal"):
        estimates = fitted[f"kalman.noise.{option}"]["coordinates"]
        assert all(item["status"] in {"fitted", "excluded"} for item in estimates)
        assert all(item["value"] < 1e-5 for item in estimates if item["status"] != "excluded")
    expected_birth = np.maximum(birth_velocity**2 / np.diag(initial_covariance)[velocity], 1e-6)
    estimates = fitted["kalman.noise.initial_velocity_diagonal"]["coordinates"]
    np.testing.assert_allclose(
        [item["value"] for item in estimates if item["status"] != "excluded"], expected_birth, rtol=1e-6, atol=1e-8
    )


@pytest.mark.parametrize(
    "geometry,kind",
    [
        (geometry, kind.value)
        for geometry in ("aabb", "obb")
        for kind in supported_motion_model_kinds(is_obb=geometry == "obb")
    ],
)
def test_identity_warps_preserve_every_existing_fit_value_and_count(kind: str, geometry: str) -> None:
    model = CalibrationModel("bytetrack", geometry, {"kalman.type": kind})
    box = np.array([60.0, 80.0, 40.0, 20.0, 0.2] if geometry == "obb" else [40.0, 70.0, 80.0, 90.0])
    boxes = np.tile(box, (10, 1))
    boxes[:, :2] += np.arange(10)[:, None] ** 2 / 100.0
    track = _track(boxes)
    expected = fit_kalman_noise([track], {0: model}, model.options)
    actual = fit_kalman_noise([track], {0: model}, model.options, camera_warps={"camera": _warps(10)})
    assert actual == expected


@pytest.mark.parametrize(
    "geometry,kind",
    [
        (geometry, kind.value)
        for geometry in ("aabb", "obb")
        for kind in supported_motion_model_kinds(is_obb=geometry == "obb")
    ],
)
@pytest.mark.parametrize("timed", [False, True])
def test_camera_jacobian_and_coordinate_bases_match_runtime_covariance(kind: str, geometry: str, timed: bool) -> None:
    model = CalibrationModel("bytetrack", geometry, {"kalman.type": kind, "kalman.variable_dt": timed})
    box = np.array([60.0, 80.0, 40.0, 20.0, 0.2] if geometry == "obb" else [40.0, 70.0, 80.0, 90.0])
    mean, covariance = model.initial_state(model.to_measurement(box))
    mean[list(model.velocity_indices)] = 0.01
    dt = 0.08 if timed else None
    coordinate_bases = model.process_coordinate_covariance_bases(mean, dt)
    position, velocity = model.process_covariance_bases(mean, dt)
    expected = position + velocity
    excluded = sorted(set(range(model.dim_x)) - set(model.measurement_indices + model.velocity_indices))
    expected[excluded] = 0
    expected[:, excluded] = 0
    np.testing.assert_allclose(coordinate_bases.sum(axis=0), expected, atol=1e-12)
    warp = np.array([[0.98, -0.1, 7.0], [0.1, 0.98, -4.0]])
    actual_mean, actual_covariance = model.motion_model.transform_states(mean[None], covariance[None], warp)
    with_jacobian, with_covariance, jacobian = model.motion_model.transform_states(
        mean[None], covariance[None], warp, return_jacobian=True
    )
    np.testing.assert_array_equal(with_jacobian, actual_mean)
    np.testing.assert_array_equal(with_covariance, actual_covariance)
    np.testing.assert_allclose(jacobian[0] @ covariance @ jacobian[0].T, actual_covariance[0], atol=1e-10)


@dataclass
class _LinearCameraModel:
    """Independent analytic 2D CV model, avoiding production covariance math."""

    dim_z: int = 2
    dim_x: int = 4
    is_obb: bool = False
    measurement_indices: tuple[int, ...] = (0, 1)
    velocity_indices: tuple[int, ...] = (2, 3)
    velocity_measurement_indices: tuple[int, ...] = (0, 1)
    position_rate: float = 1.0
    velocity_rate: float = 1.0

    def to_measurement(self, box, score=1.0, reference=None):
        return np.asarray(box).copy()

    def initial_state(self, measurement):
        return np.r_[measurement, 0.0, 0.0], np.eye(4)

    def measurement_covariance(self, measurement, score=1.0):
        return np.eye(2)

    def transition(self, dt=None):
        result = np.eye(4)
        result[:2, 2:] = np.eye(2) * (1.0 if dt is None else dt)
        return result

    def camera_transform(self, state, warp):
        jacobian = np.zeros((4, 4))
        jacobian[:2, :2] = jacobian[2:, 2:] = warp[:, :2]
        transformed = jacobian @ state
        transformed[:2] += warp[:, 2]
        return transformed, jacobian

    def process_coordinate_covariance_bases(self, mean, dt=None):
        bases = np.zeros((4, 4, 4))
        if dt is None:
            bases[np.arange(4), np.arange(4), np.arange(4)] = [self.position_rate] * 2 + [self.velocity_rate] * 2
        else:
            for axis in range(2):
                bases[axis, axis, axis] = self.position_rate * dt
                bases[axis + 2, axis, axis] = self.velocity_rate * dt**3 / 3.0
                cross = self.velocity_rate * dt**2 / 2.0
                bases[axis + 2, axis, axis + 2] = bases[axis + 2, axis + 2, axis] = cross
                bases[axis + 2, axis + 2, axis + 2] = self.velocity_rate * dt
        return bases


def _collect(track, model, *, warps, timed):
    moments = CoupledProcessNoiseMoments(4)
    birth = ScalarNoiseMoments()
    _collect_track_moments(
        track,
        model,
        variable_dt=timed,
        measurement=ScalarNoiseMoments(),
        initial_position=ScalarNoiseMoments(),
        initial_velocity=birth,
        process=moments,
        camera_warps=warps,
    )
    return moments, birth


@pytest.mark.parametrize("timed", [False, True])
def test_rotating_camera_recovers_anisotropic_position_and_velocity_diffusion(timed: bool) -> None:
    rng = np.random.default_rng(622)
    model = _LinearCameraModel(position_rate=0.05, velocity_rate=20.0) if timed else _LinearCameraModel()
    count = 10000
    scales = np.array([1.7, 0.6, 0.9, 2.5])
    intervals = rng.choice([0.03, 0.1, 0.4, 0.8], count - 1) if timed else np.ones(count - 1)
    warps = _warps(count)
    states = np.zeros((count, 4))
    states[0] = [100.0, 50.0, 3.0, 2.0]
    for index in range(1, count):
        angle = rng.uniform(-0.6, 0.6)
        scale = 1.02 if index % 2 else 1.0 / 1.02
        warps[index, :, :2] = scale * np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        warps[index, :, 2] = rng.normal(0, 10.0, 2)
        dt = float(intervals[index - 1]) if timed else None
        covariance = np.einsum("i,ijk->jk", scales, model.process_coordinate_covariance_bases(states[index - 1], dt))
        increment = np.linalg.cholesky(covariance) @ rng.standard_normal(4)
        predicted = model.transition(dt) @ states[index - 1] + increment
        states[index] = model.camera_transform(predicted, warps[index])[0]
    track = _track(states[:, :2], times=np.r_[0.0, np.cumsum(intervals)])
    moments, _ = _collect(track, model, warps=warps, timed=timed)
    estimates = moments.estimate(np.ones(4))
    assert all(item["status"] == "fitted" for item in estimates)
    np.testing.assert_allclose([item["value"] for item in estimates], scales, rtol=0.15)
    assert moments.events == count - 2
    assert moments.lag_pairs == count - 3


def test_birth_secant_is_transported_back_and_annotation_gaps_do_not_share_noise() -> None:
    model = _LinearCameraModel()
    warps = _warps(10)
    warps[:, :, :2] = np.array([[0.0, -2.0], [2.0, 0.0]])
    boxes = np.array([[10.0, 20.0], [-40.0, 26.0], [-52.0, -68.0], [136.0, -80.0]])
    track = _track(boxes)
    first, birth = _collect(track, model, warps=warps, timed=False)
    # First interval moves (3,0) in the birth frame and (0,6) after CMC.
    assert birth.total == pytest.approx(9.0)
    indices = np.array([0, 1, 2, 3, 6, 7, 8, 9])
    combined, _ = _collect(_track(np.tile(boxes, (2, 1)), indices=indices), model, warps=warps, timed=False)
    assert combined.events == 2 * first.events
    assert combined.lag_pairs == 2 * first.lag_pairs
    np.testing.assert_allclose(combined.normal, 2.0 * first.normal)
    np.testing.assert_allclose(combined.target, 2.0 * first.target)


@pytest.mark.parametrize("invalid", ["missing", "short", "nan", "singular", "shape"])
def test_camera_warps_must_cover_every_delivered_calibration_frame(invalid: str) -> None:
    track = _track(np.tile([1.0, 2.0, 3.0, 4.0], (4, 1)))
    values = _warps(4)
    if invalid == "short":
        values = values[:2]
    elif invalid == "nan":
        values[1, 0, 0] = np.nan
    elif invalid == "singular":
        values[1, :, :2] = 0.0
    elif invalid == "shape":
        values = values[:, :, :2]
    with pytest.raises(ValueError, match="Camera warps"):
        validate_camera_warps([track], {} if invalid == "missing" else {"camera": values})


def test_calibration_report_records_the_compensated_residual_model(tmp_path) -> None:
    warps = _warps(6)
    warps[1:, :, 2] = [1.0, -2.0]
    result = calibrate_kalman(
        _args(tmp_path), output_dir=tmp_path, calibration_data=_data(), camera_warps={"seq1": warps}
    )
    report = json.loads(result.report_path.read_text())
    assert report["camera_compensated"] is True
    assert any("predict then supplied CMC" in item for item in report["limitations"])
    assert not any("do not isolate tracker CMC" in item for item in report["limitations"])


def test_bound_profile_cannot_be_refitted_without_its_camera_behavior(monkeypatch, tmp_path) -> None:
    from boxmot.engine.config import trackers

    config = dict(CalibrationModel("bytetrack", "aabb", {}).options)
    config["calibration.cmc_policy"] = "auto"
    monkeypatch.setattr(trackers, "resolve_tracker_options", lambda *args, **kwargs: config)
    with pytest.raises(ValueError, match="requires its camera warps"):
        calibrate_kalman(_args(tmp_path), output_dir=tmp_path, calibration_data=_data())
