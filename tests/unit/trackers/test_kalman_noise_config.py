"""Tracker-local Kalman settings reach live filters and batched prediction."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from boxmot import KalmanNoiseConfig
from boxmot.trackers import Tracker, TrackerSpec, create_tracker
from boxmot.trackers.common.config import flatten_tracker_options
from tests.unit.trackers.test_trackers import _aabb_rows, _detections, _empty_rows, _frame, _obb_rows
from tests.unit.trackers.test_variable_frame_time import KALMAN_TRACKERS, _state, _update


def _tracker(
    name: str,
    geometry: str,
    *,
    kalman_noise: KalmanNoiseConfig | None = None,
    per_class: bool = False,
    **tracking_options: object,
) -> Tracker:
    """Resolve supplied embeddings without loading a ReID model."""
    noise_options = {} if kalman_noise is None else flatten_tracker_options({"kalman_noise": kalman_noise})
    options = tuple(sorted({"min_hits": 1, **tracking_options, **noise_options}.items()))
    return create_tracker(TrackerSpec(name, geometry=geometry, per_class=per_class, options=options))


def _covariance(track: object) -> np.ndarray:
    """Read covariance from either a stateless or stateful track wrapper."""
    _, kalman = _state(track)
    return getattr(track, "covariance", kalman.P)


@pytest.mark.parametrize("name", KALMAN_TRACKERS)
@pytest.mark.parametrize("geometry", ["aabb", "obb"])
def test_noise_settings_reach_tracks_survive_reset_and_preserve_other_owners(name: str, geometry: str) -> None:
    rows = (_obb_rows() if geometry == "obb" else _aabb_rows())[:1]
    moved = rows.copy()
    moved[:, 0] += 1.0
    if geometry == "aabb":
        moved[:, 2] += 1.0

    # Record the default forecast before a differently configured owner exists.
    reference = _tracker(name, geometry)
    _update(reference, rows, index=0)
    reference_track = reference.active_tracks[0]
    initial_mean = _state(reference_track)[0].copy()
    initial_covariance = _covariance(reference_track).copy()
    expected_output = _update(reference, moved, index=1)
    expected_mean = _state(reference_track)[0].copy()
    expected_covariance = _covariance(reference_track).copy()

    live_default = _tracker(name, geometry)
    _update(live_default, rows, index=0)
    configured = _tracker(
        name,
        geometry,
        kalman_noise=KalmanNoiseConfig(
            process_position_scale=2.0,
            process_velocity_scale=5.0,
            measurement_noise_scale=4.0,
            initial_position_scale=7.0,
            initial_velocity_scale=3.0,
        ),
    )
    _update(configured, rows, index=0)
    configured_track = configured.active_tracks[0]
    actual_mean, kalman = _state(configured_track)
    covariance = _covariance(configured_track)
    dim_z = kalman.dim_z
    config = configured.kalman_noise_config

    assert kalman.noise_config is config
    assert (
        config.process_position_scale,
        config.process_velocity_scale,
        config.measurement_noise_scale,
        config.initial_position_scale,
        config.initial_velocity_scale,
    ) == (
        2.0,
        5.0,
        4.0,
        7.0,
        3.0,
    )
    assert config.time_unit == "frames"
    np.testing.assert_array_equal(actual_mean, initial_mean)
    np.testing.assert_allclose(covariance[:dim_z, :dim_z], 7.0 * initial_covariance[:dim_z, :dim_z])
    np.testing.assert_allclose(covariance[dim_z:, dim_z:], 3.0 * initial_covariance[dim_z:, dim_z:])

    _update(configured, moved, index=1)
    actual_output = _update(live_default, moved, index=1)
    default_track = live_default.active_tracks[0]
    np.testing.assert_array_equal(_state(default_track)[0], expected_mean)
    np.testing.assert_array_equal(_covariance(default_track), expected_covariance)
    np.testing.assert_array_equal(actual_output.geometry.values, expected_output.geometry.values)
    np.testing.assert_array_equal(actual_output.track_ids, expected_output.track_ids)
    assert live_default.kalman_noise_config.is_default
    assert _state(default_track)[1].noise_config.is_default

    configured.reset()
    _update(configured, rows, index=0)
    reset_track = configured.active_tracks[0]
    assert _state(reset_track)[1].noise_config is config
    np.testing.assert_allclose(_covariance(reset_track)[:dim_z, :dim_z], 7.0 * initial_covariance[:dim_z, :dim_z])
    np.testing.assert_allclose(_covariance(reset_track)[dim_z:, dim_z:], 3.0 * initial_covariance[dim_z:, dim_z:])


@pytest.mark.parametrize("name", KALMAN_TRACKERS)
@pytest.mark.parametrize("geometry", ["aabb", "obb"])
def test_measurement_noise_changes_correction_without_changing_initial_uncertainty(name: str, geometry: str) -> None:
    rows = (_obb_rows() if geometry == "obb" else _aabb_rows())[:1]
    moved = rows.copy()
    moved[:, 0] += 1.0
    if geometry == "aabb":
        moved[:, 2] += 1.0

    initial_covariances, corrections = [], []
    for scale in (1.0, 8.0):
        tracker = _tracker(name, geometry, kalman_noise=KalmanNoiseConfig(measurement_noise_scale=scale))
        _update(tracker, rows, index=0)
        track = tracker.active_tracks[0]
        initial_covariances.append(_covariance(track).copy())
        initial_x = float(_state(track)[0][0])
        _update(tracker, moved, index=1)
        assert tracker.active_tracks == [track]
        corrections.append(float(_state(track)[0][0]) - initial_x)

    np.testing.assert_array_equal(initial_covariances[0], initial_covariances[1])
    assert 0.0 < corrections[1] < corrections[0] < 1.0


@pytest.mark.parametrize("name", KALMAN_TRACKERS)
@pytest.mark.parametrize("geometry", ["aabb", "obb"])
def test_seconds_initial_covariance_uses_reference_interval_and_independent_scales(name: str, geometry: str) -> None:
    rows = (_obb_rows() if geometry == "obb" else _aabb_rows())[:1]
    reference = _tracker(name, geometry)
    _update(reference, rows, index=0)
    reference_track = reference.active_tracks[0]
    initial_mean, kalman = _state(reference_track)
    initial_covariance = _covariance(reference_track).copy()
    dim_z = kalman.dim_z
    reference_dt = 0.05
    tracker = _tracker(
        name,
        geometry,
        variable_dt=True,
        kalman_noise=KalmanNoiseConfig(
            reference_dt_s=reference_dt,
            initial_position_scale=3.0,
            initial_velocity_scale=7.0,
            measurement_noise_scale=4.0,
        ),
    )

    for timestamp in (0.0, 20.0):
        _update(tracker, rows, index=0, timestamp_s=timestamp)
        track = tracker.active_tracks[0]
        config = _state(track)[1].noise_config
        assert config is tracker.kalman_noise_config
        assert config.time_unit == "seconds"
        assert config.reference_dt_s == reference_dt
        assert tracker._prediction_dt is None
        np.testing.assert_array_equal(_state(track)[0], initial_mean)
        np.testing.assert_allclose(_covariance(track)[:dim_z, :dim_z], 3.0 * initial_covariance[:dim_z, :dim_z])
        np.testing.assert_allclose(
            _covariance(track)[dim_z:, dim_z:], 7.0 * initial_covariance[dim_z:, dim_z:] / reference_dt**2
        )
        tracker.reset()
        assert tracker.variable_dt is True


@pytest.mark.parametrize("name", ["bytetrack", "botsort"])
@pytest.mark.parametrize("geometry", ["aabb", "obb"])
@pytest.mark.parametrize("timed", [False, True])
def test_second_frame_batch_prediction_uses_configured_owner_noise(name: str, geometry: str, timed: bool) -> None:
    tracker = _tracker(
        name,
        geometry,
        variable_dt=timed,
        kalman_noise=KalmanNoiseConfig(process_position_scale=5.0, process_velocity_scale=2.0),
    )
    rows = (_obb_rows() if geometry == "obb" else _aabb_rows())[:1]
    _update(tracker, rows, index=0, timestamp_s=0.0 if timed else None)
    track = tracker.active_tracks[0]
    mean, kalman = _state(track)
    expected_mean, expected_covariance = kalman.predict(
        mean.copy(), _covariance(track).copy(), dt=0.1 if timed else None
    )

    _update(tracker, _empty_rows(is_obb=geometry == "obb"), index=1, timestamp_s=0.1 if timed else None)

    assert tracker.kalman_filter is kalman
    np.testing.assert_array_equal(_state(track)[0], expected_mean)
    np.testing.assert_allclose(_covariance(track), expected_covariance)


@pytest.mark.parametrize("name", ["bytetrack", "botsort"])
@pytest.mark.parametrize("geometry", ["aabb", "obb"])
def test_mixed_owner_batch_preserves_each_filter_noise_configuration(name: str, geometry: str) -> None:
    rows = (_obb_rows() if geometry == "obb" else _aabb_rows())[:1]
    trackers = [
        _tracker(name, geometry, variable_dt=True, kalman_noise=KalmanNoiseConfig(process_position_scale=scale))
        for scale in (1.0, 5.0)
    ]
    tracks, expected = [], []
    for tracker in trackers:
        _update(tracker, rows, index=0, timestamp_s=0.0)
        track = tracker.active_tracks[0]
        tracks.append(track)
        mean, kalman = _state(track)
        expected.append(kalman.predict(mean.copy(), _covariance(track).copy(), dt=0.25))

    type(tracks[0]).multi_predict(tracks, dt=0.25)

    for track, (mean, covariance) in zip(tracks, expected):
        np.testing.assert_array_equal(_state(track)[0], mean)
        np.testing.assert_allclose(_covariance(track), covariance)
    assert _covariance(tracks[1])[0, 0] > _covariance(tracks[0])[0, 0]


@pytest.mark.parametrize("name", KALMAN_TRACKERS)
@pytest.mark.parametrize("geometry", ["aabb", "obb"])
@pytest.mark.parametrize("timed", [False, True])
def test_per_class_noise_reaches_independent_filters_and_preserves_pooled_fallback(name, geometry, timed) -> None:
    """Each class owns its filter; reset retains immutable calibration settings."""
    rows = np.repeat((_obb_rows() if geometry == "obb" else _aabb_rows())[:1], 3, axis=0)
    rows[:, -1] = [0, 1, 2]
    noise = KalmanNoiseConfig(
        initial_position_scale=5.0,
        initial_velocity_scale=7.0,
        by_class={
            0: KalmanNoiseConfig(initial_position_scale=11.0, initial_velocity_scale=13.0),
            1: KalmanNoiseConfig(initial_position_scale=17.0, initial_velocity_scale=19.0),
        },
    )
    baseline = _tracker(name, geometry, per_class=True, variable_dt=timed)
    _update(baseline, rows, index=0, timestamp_s=0.0 if timed else None)
    initial = _covariance(baseline.get_class_tracks(2)[0]).copy()
    tracker = _tracker(name, geometry, per_class=True, kalman_noise=noise, variable_dt=timed)
    pooled = tracker.kalman_noise_config
    original_filter = getattr(tracker, "kalman_filter", None)

    for _ in range(2):
        _update(tracker, rows, index=0, timestamp_s=0.0 if timed else None)
        filters = []
        for class_id, position, velocity in ((0, 11.0, 13.0), (1, 17.0, 19.0), (2, 5.0, 7.0)):
            track = tracker.get_class_tracks(class_id)[0]
            kalman = _state(track)[1]
            filters.append(kalman)
            assert kalman.noise_config is pooled.for_class(class_id)
            split = kalman.dim_z
            np.testing.assert_allclose(_covariance(track)[:split, :split], initial[:split, :split] * position)
            np.testing.assert_allclose(_covariance(track)[split:, split:], initial[split:, split:] * velocity)
        assert len({id(value) for value in filters}) == 3
        assert tracker.kalman_noise_config is pooled
        assert getattr(tracker, "kalman_filter", None) is original_filter
        _update(tracker, _empty_rows(is_obb=geometry == "obb"), index=1, timestamp_s=0.1 if timed else None)
        assert [value.noise_config for value in filters] == [pooled.for_class(value) for value in (0, 1, 2)]
        tracker.reset()
        original_filter = getattr(tracker, "kalman_filter", None)
        assert tracker.kalman_noise_config is pooled


@pytest.mark.parametrize("name", ["bytetrack", "botsort"])
def test_per_class_noise_restores_owner_after_a_kernel_error(name, monkeypatch) -> None:
    noise = KalmanNoiseConfig(by_class={0: KalmanNoiseConfig(measurement_noise_scale=8.0)})
    tracker = _tracker(name, "aabb", per_class=True, kalman_noise=noise)
    pooled = tracker.kalman_noise_config
    original_filter = tracker.kalman_filter

    def fail(**kwargs):
        assert tracker.kalman_noise_config is pooled.for_class(0)
        assert tracker.kalman_filter is not original_filter
        raise RuntimeError("fixture kernel failure")

    monkeypatch.setattr(tracker, "_track_detections", fail)
    with pytest.raises(RuntimeError, match="fixture kernel failure"):
        _update(tracker, _aabb_rows()[:1], index=0)
    assert tracker.kalman_noise_config is pooled
    assert tracker.kalman_filter is original_filter


@pytest.mark.parametrize("name", KALMAN_TRACKERS)
def test_class_noise_selects_canonical_ids_before_float32_kernel_encoding(name) -> None:
    class_id = 2**40 + 3
    noise = KalmanNoiseConfig(by_class={class_id: KalmanNoiseConfig(measurement_noise_scale=9.0)})
    tracker = _tracker(name, "aabb", per_class=True, kalman_noise=noise)
    rows = np.repeat(_aabb_rows()[:1], 2, axis=0)
    detections = _detections(rows, sample_id="large-class", embeddings=tracker.requirements.embeddings)
    detections = replace(detections, class_ids=torch.tensor([class_id, 0], dtype=torch.int64))
    tracker.update(detections, _frame("large-class", 0))
    calibrated = tracker.get_class_tracks(class_id)[0]
    fallback = tracker.get_class_tracks(0)[0]
    assert _state(calibrated)[1].noise_config is tracker.kalman_noise_config.for_class(class_id)
    assert _state(calibrated)[1].noise_config.measurement_noise_scale == 9.0
    assert _state(fallback)[1].noise_config is tracker.kalman_noise_config


@pytest.mark.parametrize("name", KALMAN_TRACKERS)
def test_box_class_noise_requires_class_separation(name) -> None:
    config = KalmanNoiseConfig(by_class={0: KalmanNoiseConfig()})
    with pytest.raises(ValueError, match="requires per_class=True"):
        _tracker(name, "aabb", kalman_noise=config)
