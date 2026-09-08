"""Supervised residual collection identifies simulated process diffusion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from boxmot.engine.tuning.kalman import _collect_track_moments
from boxmot.engine.tuning.kalman_data import CalibrationTrack
from boxmot.motion.kalman_filters.fitting import ProcessNoiseMoments, ScalarNoiseMoments


@dataclass
class _OneDimensionalModel:
    """Independent CV reference with analytically discretized diffusion."""

    position_rate: float = 1.0
    velocity_rate: float = 1.0
    dim_z: int = 1
    dim_x: int = 2
    measurement_indices: tuple[int, ...] = (0,)
    velocity_indices: tuple[int, ...] = (1,)
    velocity_measurement_indices: tuple[int, ...] = (0,)

    def to_measurement(self, box: np.ndarray, score: float = 1.0, reference: np.ndarray | None = None) -> np.ndarray:
        del score, reference
        return np.asarray(box).copy()

    def initial_state(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.r_[measurement, 0.0], np.eye(2)

    def measurement_covariance(self, measurement: np.ndarray, score: float = 1.0) -> np.ndarray:
        del measurement, score
        return np.ones((1, 1))

    def transition(self, dt: float | None = None) -> np.ndarray:
        return np.array([[1.0, 1.0 if dt is None else dt], [0.0, 1.0]])

    def process_covariance_bases(self, mean: np.ndarray, dt: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        del mean
        if dt is None:
            return np.diag([self.position_rate, 0.0]), np.diag([0.0, self.velocity_rate])
        position = self.position_rate * np.array([[dt, 0.0], [0.0, 0.0]])
        velocity = self.velocity_rate * np.array([[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]])
        return position, velocity


def _track(positions: np.ndarray, times: np.ndarray, indices: np.ndarray | None = None) -> CalibrationTrack:
    """Provide perfect GT and explicit detector misses to isolate process fit."""
    indices = np.arange(len(positions)) if indices is None else indices
    return CalibrationTrack(
        sequence_id="simulation",
        track_id=1,
        class_id=0,
        frame_indices=indices,
        timestamps_s=times,
        gt_boxes=positions.reshape(-1, 1),
        detection_boxes=np.full((len(positions), 1), np.nan),
        scores=np.full(len(positions), np.nan),
    )


def _collect(track: CalibrationTrack, model: _OneDimensionalModel, *, variable_dt: bool) -> ProcessNoiseMoments:
    moments = ProcessNoiseMoments()
    _collect_track_moments(
        track,
        model,
        variable_dt=variable_dt,
        measurement=ScalarNoiseMoments(),
        initial_position=ScalarNoiseMoments(),
        initial_velocity=ScalarNoiseMoments(),
        process=moments,
    )
    return moments


@pytest.mark.parametrize("timing", ["fixed", "uniform_seconds", "irregular_seconds"])
def test_actual_gt_collection_recovers_simulated_position_and_velocity_diffusion(timing: str) -> None:
    rng = np.random.default_rng(2508)
    count = 16000
    position_scale, velocity_scale = 1.7, 0.6
    variable_dt = timing != "fixed"
    if variable_dt:
        model = _OneDimensionalModel(position_rate=0.05, velocity_rate=20.0)
        intervals = (
            rng.choice([1.0 / 30.0, 0.1, 0.2, 0.37], size=count - 1)
            if timing == "irregular_seconds"
            else np.full(count - 1, 0.1)
        )
    else:
        model = _OneDimensionalModel()
        intervals = np.ones(count - 1)
    covariance = np.stack(
        [
            position_scale * position + velocity_scale * velocity
            for position, velocity in (
                model.process_covariance_bases(np.zeros(2), dt=float(dt) if variable_dt else None) for dt in intervals
            )
        ]
    )
    increments = np.einsum("nij,nj->ni", np.linalg.cholesky(covariance), rng.standard_normal((count - 1, 2)))
    latent_velocities = np.r_[2.0, 2.0 + np.cumsum(increments[:, 1])]
    positions = np.r_[10.0, 10.0 + np.cumsum(intervals * latent_velocities[:-1] + increments[:, 0])]
    times = np.r_[0.0, np.cumsum(intervals)]

    moments = _collect(_track(positions, times), model, variable_dt=variable_dt)
    position, velocity = moments.estimate((1.0, 1.0))

    # This checks the full measurement→secant→projected variance/lag pipeline.
    # Finite simulated trajectories have sampling error and correlated moments.
    assert position["value"] == pytest.approx(position_scale, rel=0.15)
    assert velocity["value"] == pytest.approx(velocity_scale, rel=0.15)
    assert moments.events == count - 2
    assert moments.lag_pairs == count - 3


def test_gt_annotation_gap_never_adds_a_cross_covariance_between_segments() -> None:
    model = _OneDimensionalModel(position_rate=0.05, velocity_rate=20.0)
    times = np.array([0.0, 0.03, 0.1, 0.2, 0.5, 0.7, 0.8, 1.0])
    indices = np.array([0, 1, 2, 3, 7, 8, 9, 10])
    positions = np.array([10.0, 10.1, 10.5, 10.7, 40.0, 40.1, 40.4, 41.0])

    combined = _collect(_track(positions, times, indices), model, variable_dt=True)
    first = _collect(_track(positions[:4], times[:4], indices[:4]), model, variable_dt=True)
    second = _collect(_track(positions[4:], times[4:], indices[4:]), model, variable_dt=True)

    assert combined.events == 4
    assert combined.lag_pairs == 2
    np.testing.assert_allclose(combined.normal, first.normal + second.normal)
    np.testing.assert_allclose(combined.target, first.target + second.target)


def test_fixed_step_collection_ignores_capture_interval_lengths() -> None:
    model = _OneDimensionalModel()
    positions = np.array([10.0, 10.1, 10.5, 10.7, 11.2])
    uniform = np.arange(len(positions), dtype=float)
    irregular = np.array([0.0, 0.03, 0.1, 0.4, 0.7])

    first = _collect(_track(positions, uniform), model, variable_dt=False)
    second = _collect(_track(positions, irregular), model, variable_dt=False)

    np.testing.assert_array_equal(first.normal, second.normal)
    np.testing.assert_array_equal(first.target, second.target)


def test_random_walk_coordinate_without_a_velocity_uses_only_current_interval_noise() -> None:
    class RandomWalkModel(_OneDimensionalModel):
        """Reference coordinate with no derivative state, like XYSR aspect ratio."""

        def __init__(self) -> None:
            super().__init__(dim_x=1, velocity_indices=(), velocity_measurement_indices=())

        def transition(self, dt: float | None = None) -> np.ndarray:
            del dt
            return np.ones((1, 1))

        def process_covariance_bases(self, mean: np.ndarray, dt: float | None = None) -> tuple[np.ndarray, np.ndarray]:
            del mean
            return np.array([[1.0 if dt is None else dt]]), np.zeros((1, 1))

    times = np.array([0.0, 0.02, 0.12, 0.32, 0.39])
    positions = np.array([10.0, 10.2, 10.3, 10.0, 10.4])

    moments = _collect(_track(positions, times), RandomWalkModel(), variable_dt=True)

    # First-difference residuals use only their own interval. There is no
    # artificial past-noise contribution or covariance between independent steps.
    np.testing.assert_array_equal(moments.normal, np.array([[3.0, 0.0], [0.0, 0.0]]))
    expected = np.sum(np.diff(positions)[1:] ** 2 / np.diff(times)[1:])
    np.testing.assert_allclose(moments.target, [expected, 0.0])
    assert moments.events == 3
    assert moments.lag_pairs == 0
