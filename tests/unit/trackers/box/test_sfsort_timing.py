"""SFSORT retains fixed observation updates and rejects variable timing settings."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from boxmot import SFSORT, KalmanConfig, SFSORTConfig
from boxmot.structures import Tracks
from boxmot.trackers import TrackerSpec, create_tracker
from tests.unit.trackers.test_trackers import _detections, _frame


def _tracker(*, is_obb: bool) -> SFSORT:
    """Retain missing identities through both confidence association passes."""
    return SFSORT(
        SFSORTConfig(
            frame_width=128,
            frame_height=96,
            high_th=0.5,
            new_track_th=0.5,
            match_th_second=0.5,
            central_timeout=5,
            marginal_timeout=5,
        ),
        is_obb=is_obb,
    )


def _rows(*, is_obb: bool, angle: float = 0.0, confidence: float = 0.95) -> np.ndarray:
    """Use overlapping boxes so angular changes preserve the matched identity."""
    box = [64, 48, 40, 20, angle] if is_obb else [44, 38, 84, 58]
    return np.array([[*box, confidence, 0]], dtype=np.float64)


@pytest.mark.parametrize("variable_dt", [False, True])
def test_sfsort_rejects_kalman_configuration(variable_dt: bool) -> None:
    """Timing cannot introduce a Kalman component into the observation-only tracker."""
    with pytest.raises(TypeError, match="does not accept kalman"):
        SFSORT(kalman=KalmanConfig(variable_dt=variable_dt))


def test_factory_rejects_sfsort_variable_timing() -> None:
    """Factory construction must reject the unsupported timestamp-dependent mode."""
    with pytest.raises(ValueError, match="kalman"):
        create_tracker(TrackerSpec("sfsort", options=(("kalman.variable_dt", True),)))


@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("confidence", [0.95, 0.3])
@pytest.mark.parametrize("timing", ["explicit", "frame"])
def test_capture_timestamps_do_not_change_observed_boxes_or_obb_smoothing(
    is_obb: bool, confidence: float, timing: str
) -> None:
    """Irregular and reversed metadata leave fixed updates and recovery unchanged."""
    baseline = _tracker(is_obb=is_obb)
    stamped = _tracker(is_obb=is_obb)
    for index, (angle, timestamp) in enumerate(((0.0, 10.0), (0.2, 10.1), (None, 20.0), (0.4, 9.0))):
        rows = _rows(is_obb=is_obb, angle=angle or 0.0, confidence=0.95 if index == 0 else confidence)
        if angle is None:
            rows = rows[:0]
        if timing == "explicit":
            expected = baseline.update(rows)
            actual = stamped.update(rows, timestamp_s=timestamp)
        else:
            sample_id = f"sequence/{index}"
            detections = _detections(rows, sample_id=sample_id)
            frame = _frame(sample_id, index)
            expected_tracks = baseline.update(detections, frame)
            actual_tracks = stamped.update(detections, replace(frame, timestamp_s=timestamp))
            assert isinstance(actual_tracks, Tracks)
            expected = expected_tracks.geometry.values.numpy()
            actual = actual_tracks.geometry.values.numpy()
        np.testing.assert_array_equal(actual, expected)
        assert stamped.supports_variable_dt is False
        assert stamped.variable_dt is False
        assert stamped.requirements.timestamp is False
        assert stamped._prediction_dt is None
        assert stamped._last_timestamp_s is None

    assert len(stamped.active_tracks) == 1
    assert stamped.active_tracks[0].id == baseline.active_tracks[0].id
    np.testing.assert_array_equal(
        stamped.active_tracks[0].history_observations,
        baseline.active_tracks[0].history_observations,
    )
    if is_obb:
        assert 0.0 < stamped.active_tracks[0].bbox[4] < 0.4
