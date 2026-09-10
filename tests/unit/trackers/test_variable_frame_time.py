"""Trackers own capture timestamps and derive elapsed prediction intervals."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from boxmot.structures import Tracks
from boxmot.trackers import Tracker, TrackerSpec, create_tracker
from boxmot.trackers.bytetrack.native import NativeByteTrackTracker
from boxmot.trackers.common.motion.kalman_filters.base import BaseKalmanFilter
from tests.unit.native.trackers.test_native_bytetrack import _FakeLibrary
from tests.unit.trackers.test_trackers import _aabb_rows, _detections, _empty_rows, _frame, _obb_rows

KALMAN_TRACKERS = (
    "bytetrack",
    "ocsort",
    "botsort",
    "strongsort",
    "deepocsort",
    "hybridsort",
    "boosttrack",
    "occluboost",
)


def _update(
    tracker: Tracker,
    rows: np.ndarray,
    *,
    index: int,
    packed: bool = False,
    timestamp_s: object = None,
    timing: str = "frame",
) -> Tracks | np.ndarray:
    """Supply required perception data without loading external models."""
    sample_id = f"sequence/{index}"
    detections = (
        rows
        if packed
        else _detections(
            rows,
            sample_id=sample_id,
            embeddings=tracker.requirements.embeddings,
            masks=tracker.requirements.masks,
        )
    )
    frame = _frame(sample_id, index) if tracker.requirements.frame or timing == "frame" else None
    if timing == "frame":
        frame = replace(frame, timestamp_s=timestamp_s)
        return tracker.update(detections, frame)
    return tracker.update(detections, frame, timestamp_s=timestamp_s)


def _initialized_tracker(
    name: str, geometry: str, *, packed: bool = False, timing: str = "frame", timed: bool = True
) -> Tracker:
    """Confirm one track before inspecting a missing-frame prediction."""
    tracker = create_tracker(TrackerSpec(name, geometry=geometry, options=(("min_hits", 1), ("variable_dt", timed))))
    rows = (_obb_rows() if geometry == "obb" else _aabb_rows())[:1]
    for index in range(4):
        output = _update(
            tracker,
            rows,
            index=index,
            packed=packed,
            timestamp_s=10.0 + index / 10.0 if timed else None,
            timing=timing,
        )
    assert len(output) == 1
    assert len(tracker.active_tracks) == 1
    return tracker


def _state(track: object) -> tuple[np.ndarray, BaseKalmanFilter]:
    """Expose the state held by either stateless or stateful track filters."""
    kf = getattr(track, "kalman_filter", None)
    if kf is None:
        kf = track.kf
    mean = getattr(track, "mean", None)
    return (kf.x if mean is None else mean).reshape(-1), kf


@pytest.mark.parametrize("name", KALMAN_TRACKERS)
@pytest.mark.parametrize("geometry", ["aabb", "obb"])
def test_frame_timestamp_drives_prediction_for_every_kalman_tracker(name: str, geometry: str) -> None:
    tracker = _initialized_tracker(name, geometry)
    track = tracker.active_tracks[0]
    mean, kf = _state(track)
    mean[kf.dim_z] = 12.0
    if geometry == "obb":
        mean[-1] = 0.2
    before = mean.copy()

    output = _update(tracker, _empty_rows(is_obb=geometry == "obb"), index=4, timestamp_s=10.55)
    predicted, _ = _state(track)

    assert tracker.supports_variable_dt is True
    assert isinstance(output, Tracks)
    assert output.is_obb is (geometry == "obb")
    assert predicted[0] == pytest.approx(before[0] + 3.0)
    if geometry == "obb":
        assert predicted[4] == pytest.approx(before[4] + 0.05)


@pytest.mark.parametrize("name", KALMAN_TRACKERS)
@pytest.mark.parametrize("geometry", ["aabb", "obb"])
def test_irregular_intervals_preserve_motion_and_identity_through_missing_detections(name: str, geometry: str) -> None:
    tracker = _initialized_tracker(name, geometry)
    track = tracker.active_tracks[0]
    mean, kalman = _state(track)
    mean[kalman.dim_z] = 30.0
    initial_x = float(mean[0])
    empty = _empty_rows(is_obb=geometry == "obb")
    previous_variance = float(getattr(track, "covariance", kalman.P)[0, 0])
    timestamp = 10.3
    displacement = 0.0

    for index, interval in enumerate((1.0 / 30.0, 0.1), start=4):
        timestamp += interval
        displacement += 30.0 * interval
        _update(tracker, empty, index=index, timestamp_s=timestamp)
        covariance = getattr(track, "covariance", kalman.P)
        assert tracker._prediction_dt == pytest.approx(interval)
        assert _state(track)[0][0] == pytest.approx(initial_x + displacement)
        assert covariance[0, 0] > previous_variance
        assert np.isfinite(covariance).all()
        assert np.linalg.eigvalsh(covariance).min() >= -1e-7
        previous_variance = float(covariance[0, 0])

    timestamp += 1.0 / 30.0
    rows = (_obb_rows() if geometry == "obb" else _aabb_rows())[:1].copy()
    rows[:, 0] += 5.0
    if geometry == "aabb":
        rows[:, 2] += 5.0
    result = _update(tracker, rows, index=6, timestamp_s=timestamp)

    assert tracker.active_tracks == [track]
    assert len(result) == 1
    assert _state(track)[0][0] == pytest.approx(initial_x + 5.0, abs=0.5)
    covariance = getattr(track, "covariance", kalman.P)
    assert np.isfinite(covariance).all()
    assert np.linalg.eigvalsh(covariance).min() >= -1e-7
    assert covariance[0, 0] < previous_variance


@pytest.mark.parametrize("name", ["bytetrack", "ocsort"])
@pytest.mark.parametrize("geometry", ["aabb", "obb"])
@pytest.mark.parametrize("packed", [False, True])
def test_explicit_capture_timestamps_drive_prediction_and_preserve_input_format(
    name: str, geometry: str, packed: bool
) -> None:
    tracker = _initialized_tracker(name, geometry, packed=packed, timing="explicit")
    track = tracker.active_tracks[0]
    mean, kf = _state(track)
    mean[kf.dim_z] = 12.0
    initial_x = float(mean[0])
    empty = _empty_rows(is_obb=geometry == "obb")

    output = _update(tracker, empty, index=4, packed=packed, timestamp_s=10.55, timing="explicit")
    assert _state(track)[0][0] == pytest.approx(initial_x + 3.0)
    _update(tracker, empty, index=5, packed=packed, timestamp_s=11.55, timing="explicit")

    assert _state(track)[0][0] == pytest.approx(initial_x + 15.0)
    if packed:
        assert type(output) is np.ndarray
        assert output.dtype == np.float64
        assert output.flags.c_contiguous
        assert output.shape[1] == (9 if geometry == "obb" else 8)
    else:
        assert isinstance(output, Tracks)


@pytest.mark.parametrize("name", ["bytetrack", "ocsort"])
@pytest.mark.parametrize("timestamp", [10.3, 10.0, np.nan, np.inf, True, "10.55", [10.55]])
def test_invalid_timestamp_rejects_update_before_mutating_live_tracks(name: str, timestamp: object) -> None:
    tracker = _initialized_tracker(name, "aabb", timing="explicit")
    track = tracker.active_tracks[0]
    state_before = _state(track)[0].copy()
    frame_count = tracker.frame_count
    previous_dt = tracker._prediction_dt

    with pytest.raises(ValueError, match="timestamp"):
        _update(tracker, _aabb_rows()[:1], index=4, timestamp_s=timestamp, timing="explicit")

    np.testing.assert_array_equal(_state(track)[0], state_before)
    assert tracker.frame_count == frame_count
    assert tracker._prediction_dt == previous_dt
    assert tracker.active_tracks == [track]
    _update(tracker, _empty_rows(is_obb=False), index=4, timestamp_s=10.55, timing="explicit")
    assert tracker._prediction_dt == pytest.approx(0.25)


def test_class_separated_tracks_share_one_elapsed_interval() -> None:
    tracker = create_tracker(
        TrackerSpec("bytetrack", per_class=True, class_ids=(0, 65), options=(("variable_dt", True),))
    )
    _update(tracker, _aabb_rows(), index=0, timestamp_s=10.0)
    tracks = tracker.active_tracks.copy()
    assert len(tracks) == 2
    initial_x = []
    for track in tracks:
        mean, kf = _state(track)
        mean[kf.dim_z] = 12.0
        initial_x.append(float(mean[0]))

    _update(tracker, _empty_rows(is_obb=False), index=1, timestamp_s=10.25)

    np.testing.assert_allclose([_state(track)[0][0] for track in tracks], np.asarray(initial_x) + 3.0)
    assert tracker.frame_count == 2


@pytest.mark.parametrize("name", ["sfsort", "maf_hda"])
def test_trackers_without_timed_motion_reject_enabling_variable_dt(name: str) -> None:
    with pytest.raises(ValueError, match="variable_dt"):
        create_tracker(TrackerSpec(name, options=(("variable_dt", True),)))


def test_native_tracker_ignores_frame_and_explicit_timestamp_metadata() -> None:
    library = _FakeLibrary()
    tracker = NativeByteTrackTracker(library=library)
    try:
        assert tracker.variable_dt is False
        output = tracker.update(_aabb_rows(), timestamp_s=10.25)
        assert output.shape == (0, 8)
        output = tracker.update(_aabb_rows(), replace(_frame("one", 0), timestamp_s=10.0))
        assert output.shape == (0, 8)
        output = tracker.update(_aabb_rows(), replace(_frame("two", 1), timestamp_s=9.0))
        assert output.shape == (0, 8)
    finally:
        tracker.close()


@pytest.mark.parametrize("name", ["bytetrack", "ocsort"])
@pytest.mark.parametrize("timing", ["frame", "explicit"])
def test_first_timestamp_anchors_clock_and_reset_starts_a_new_sequence(name: str, timing: str) -> None:
    tracker = create_tracker(TrackerSpec(name, options=(("variable_dt", True),)))
    empty = _empty_rows(is_obb=False)

    _update(tracker, empty, index=0, timestamp_s=0.0, timing=timing)
    assert tracker._prediction_dt is None
    _update(tracker, empty, index=1, timestamp_s=0.25, timing=timing)
    assert tracker._prediction_dt == pytest.approx(0.25)

    tracker.reset()
    assert tracker.variable_dt is True
    _update(tracker, empty, index=0, timestamp_s=5.0, timing=timing)
    assert tracker._prediction_dt is None
    _update(tracker, empty, index=1, timestamp_s=5.5, timing=timing)
    assert tracker._prediction_dt == pytest.approx(0.5)


@pytest.mark.parametrize("name", ["bytetrack", "ocsort"])
@pytest.mark.parametrize("initialized", [False, True])
def test_variable_dt_requires_timestamp_on_every_frame(name: str, initialized: bool) -> None:
    tracker = (
        _initialized_tracker(name, "aabb")
        if initialized
        else create_tracker(TrackerSpec(name, options=(("variable_dt", True),)))
    )
    frame_count = tracker.frame_count

    with pytest.raises(ValueError, match="timestamp"):
        _update(tracker, _empty_rows(is_obb=False), index=4)

    assert tracker.frame_count == frame_count
    tracker.reset()
    assert tracker.variable_dt is True
    with pytest.raises(ValueError, match="timestamp"):
        _update(tracker, _empty_rows(is_obb=False), index=0)
    _update(tracker, _aabb_rows()[:1], index=0, timestamp_s=10.0)
    assert tracker._prediction_dt is None


@pytest.mark.parametrize("name", ["bytetrack", "ocsort"])
@pytest.mark.parametrize("packed", [False, True])
def test_untimed_sequence_retains_one_step_prediction(name: str, packed: bool) -> None:
    tracker = _initialized_tracker(name, "aabb", packed=packed, timed=False, timing="explicit")
    track = tracker.active_tracks[0]
    mean, kf = _state(track)
    mean[kf.dim_z] = 12.0
    initial_x = float(mean[0])

    _update(tracker, _empty_rows(is_obb=False), index=4, packed=packed, timing="explicit")

    assert _state(track)[0][0] == pytest.approx(initial_x + 12.0)
    assert tracker._prediction_dt is None


@pytest.mark.parametrize("name", ["bytetrack", "ocsort"])
def test_frame_and_explicit_timestamps_cannot_both_supply_time(name: str) -> None:
    tracker = create_tracker(TrackerSpec(name, options=(("variable_dt", True),)))
    frame = replace(_frame("one", 0), timestamp_s=10.0)

    with pytest.raises(ValueError, match="timestamp"):
        tracker.update(_aabb_rows(), frame, timestamp_s=10.0)

    assert tracker.frame_count == 0
    tracker.update(_aabb_rows(), frame)
    assert tracker._prediction_dt is None


@pytest.mark.parametrize("name", ["bytetrack", "ocsort"])
def test_failed_input_validation_does_not_consume_capture_timestamp(name: str) -> None:
    tracker = _initialized_tracker(name, "aabb", timing="explicit")
    invalid = _aabb_rows().copy()
    invalid[0, 4] = 2.0

    with pytest.raises(ValueError, match="range"):
        tracker.update(invalid, timestamp_s=10.55)

    _update(tracker, _empty_rows(is_obb=False), index=4, timestamp_s=10.55, timing="explicit")
    assert tracker._prediction_dt == pytest.approx(0.25)


def test_failed_tracking_kernel_does_not_consume_capture_timestamp(monkeypatch: pytest.MonkeyPatch) -> None:
    tracker = _initialized_tracker("bytetrack", "aabb", timing="explicit")

    def fail_kernel(*args: object, **kwargs: object) -> None:
        raise RuntimeError("failed before prediction")

    with monkeypatch.context() as patch:
        patch.setattr(tracker, "_track_detections", fail_kernel)
        with pytest.raises(RuntimeError, match="before prediction"):
            _update(tracker, _aabb_rows()[:1], index=4, timestamp_s=10.55, timing="explicit")

    _update(tracker, _empty_rows(is_obb=False), index=4, timestamp_s=10.55, timing="explicit")
    assert tracker._prediction_dt == pytest.approx(0.25)


@pytest.mark.parametrize("name", ["sfsort", "maf_hda"])
def test_trackers_without_timed_motion_ignore_frame_timestamps(name: str) -> None:
    tracker = create_tracker(TrackerSpec(name))

    _update(tracker, _aabb_rows()[:1], index=0, timestamp_s=10.0)
    output = _update(tracker, _aabb_rows()[:1], index=1, timestamp_s=9.0)

    assert isinstance(output, Tracks)
    assert tracker._prediction_dt is None


@pytest.mark.parametrize("timestamp", [np.nan, np.inf, True, "10.0"])
def test_first_timestamp_must_be_finite_real_value(timestamp: object) -> None:
    tracker = create_tracker(TrackerSpec("bytetrack", options=(("variable_dt", True),)))

    with pytest.raises(ValueError, match="timestamp"):
        tracker.update(_aabb_rows(), timestamp_s=timestamp)

    assert tracker.frame_count == 0
    tracker.update(_aabb_rows(), timestamp_s=0.0)
    assert tracker._prediction_dt is None


def test_frame_timestamps_do_not_require_pixel_conversion_for_iou_tracking(monkeypatch: pytest.MonkeyPatch) -> None:
    tracker = create_tracker(TrackerSpec("bytetrack", options=(("asso_func", "iou"), ("variable_dt", True))))
    assert tracker.requirements.frame is False

    def reject_pixel_conversion(*args: object, **kwargs: object) -> None:
        raise AssertionError("Timestamp metadata must not trigger image conversion")

    monkeypatch.setattr(tracker, "_frame_to_bgr", reject_pixel_conversion)
    _update(tracker, _aabb_rows()[:1], index=0, timestamp_s=10.0)
    output = _update(tracker, _aabb_rows()[:1], index=1, timestamp_s=10.25)

    assert isinstance(output, Tracks)
    assert len(output) == 1
    assert tracker._prediction_dt == pytest.approx(0.25)


@pytest.mark.parametrize("name", KALMAN_TRACKERS)
@pytest.mark.parametrize("geometry", ["aabb", "obb"])
def test_fixed_mode_has_exact_state_and_output_parity_regardless_of_frame_timestamps(name: str, geometry: str) -> None:
    """Capture metadata must preserve the model used by existing tuned configs."""
    spec = TrackerSpec(name, geometry=geometry, options=(("min_hits", 1),))
    reference, with_metadata = create_tracker(spec), create_tracker(spec)
    assert reference.variable_dt is with_metadata.variable_dt is False

    for index, timestamp in enumerate((0.0, 0.033, 0.033, None, 0.01, 0.133)):
        rows = (_obb_rows() if geometry == "obb" else _aabb_rows())[:1].copy()
        rows[:, 0] += index
        if geometry == "aabb":
            rows[:, 2] += index
        if index == 3:
            rows = rows[:0]
        expected = _update(reference, rows, index=index)
        actual = _update(with_metadata, rows, index=index, timestamp_s=timestamp)
        expected_rows = expected.to_obb_rows() if geometry == "obb" else expected.to_aabb_rows()
        actual_rows = actual.to_obb_rows() if geometry == "obb" else actual.to_aabb_rows()
        np.testing.assert_array_equal(actual_rows, expected_rows)
        assert with_metadata._prediction_dt is None
        assert len(reference.active_tracks) == len(with_metadata.active_tracks)
        for expected_track, actual_track in zip(reference.active_tracks, with_metadata.active_tracks):
            expected_mean, expected_kf = _state(expected_track)
            actual_mean, actual_kf = _state(actual_track)
            np.testing.assert_array_equal(actual_mean, expected_mean)
            expected_covariance = getattr(expected_track, "covariance", expected_kf.P)
            actual_covariance = getattr(actual_track, "covariance", actual_kf.P)
            np.testing.assert_array_equal(actual_covariance, expected_covariance)


@pytest.mark.parametrize("timestamp", [None, 10.0, 9.0, np.nan, True])
def test_fixed_mode_ignores_explicit_timestamp_metadata(timestamp: object) -> None:
    tracker = _initialized_tracker("bytetrack", "aabb", timed=False, timing="explicit")
    track = tracker.active_tracks[0]
    mean, kf = _state(track)
    mean[kf.dim_z] = 12.0
    initial_x = float(mean[0])

    _update(tracker, _empty_rows(is_obb=False), index=4, timestamp_s=timestamp, timing="explicit")

    assert _state(track)[0][0] == pytest.approx(initial_x + 12.0)
    assert tracker._prediction_dt is None


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_variable_dt_configuration_requires_boolean(value: object) -> None:
    with pytest.raises(TypeError, match="variable_dt"):
        create_tracker(TrackerSpec("bytetrack", options=(("variable_dt", value),)))


def test_native_variable_dt_rejected_before_allocating_handle() -> None:
    library = _FakeLibrary()

    with pytest.raises(ValueError, match="variable_dt"):
        NativeByteTrackTracker({"variable_dt": True}, library=library)

    assert library.calls == []
