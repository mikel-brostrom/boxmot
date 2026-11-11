"""Elapsed capture time drives Gaussian motion and masked tracklet recovery."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from boxmot import KalmanConfig, KalmanNoiseConfig, MafHda, MafHdaConfig
from boxmot.trackers.maf_hda.association import INITIAL_COVARIANCE, predict_covariance
from tests.unit.trackers.multimodal.test_maf_hda_lifecycle import _sample


@pytest.mark.parametrize("per_class", [False, True])
def test_irregular_intervals_preserve_velocity_and_recover_masks_across_gaps(per_class: bool) -> None:
    """A large elapsed gap still counts as two frames and recovers the timed trajectory."""
    tracker = MafHda(
        MafHdaConfig(max_age=3, velocity_alpha=0.0, s2ta_mode="motion", t2ta_mode="motion"),
        kalman=KalmanConfig(variable_dt=True),
        per_class=per_class,
    )
    initial_ids = None
    for frame, (timestamp, x) in enumerate(((10.0, 10), (10.1, 12), (10.5, 20))):
        detections, image = _sample(frame, [[x, 20, x + 8, 40], [x, 60, x + 8, 80]], classes=[9, 2])
        result = tracker.update(detections, image, timestamp_s=timestamp)
        if initial_ids is None:
            initial_ids = result.track_ids.tolist()
        assert result.track_ids.tolist() == initial_ids
    for track in tracker.active_tracks:
        np.testing.assert_allclose(track.velocity, [20.0, 0.0], atol=1e-12)
        assert track.first_timestamp_s == 10.0
        assert track.last_timestamp_s == 10.5
    missing = tracker.update(*_sample(3, []), timestamp_s=10.8)
    assert len(missing) == 0
    detections, image = _sample(4, [[50, 20, 58, 40], [50, 60, 58, 80]], classes=[9, 2])
    recovered = tracker.update(detections, image, timestamp_s=12.0)
    assert recovered.track_ids.tolist() == initial_ids
    assert tracker.frame_count == 5
    for row, index in enumerate(recovered.detection_indices.tolist()):
        torch.testing.assert_close(recovered.masks.values[row], detections.masks.values[index])
    for track in tracker.active_tracks:
        np.testing.assert_allclose(track.velocity, [20.0, 0.0], atol=1e-12)
        assert track.first_timestamp_s == 10.0
    tracker.reset()
    tracker.update(*_sample(0, [[10, 20, 18, 40]]), timestamp_s=1.0)
    assert tracker.active_tracks[0].first_timestamp_s == 1.0


def test_recovery_uses_candidate_birth_time_when_confirmation_is_delayed() -> None:
    """An old trajectory is compared to the candidate's first geometry and timestamp."""
    tracker = MafHda(
        MafHdaConfig(max_age=3, min_hits=2, velocity_alpha=0.0, s2ta_mode="motion", t2ta_mode="motion"),
        kalman=KalmanConfig(variable_dt=True),
    )
    tracker.update(*_sample(0, [[10, 20, 18, 40]]), timestamp_s=10.0)
    first = tracker.update(*_sample(1, [[12, 20, 20, 40]]), timestamp_s=10.1)
    tracker.update(*_sample(2, []), timestamp_s=10.4)
    unconfirmed = tracker.update(*_sample(3, [[50, 20, 58, 40]]), timestamp_s=12.0)
    assert len(unconfirmed) == 0
    recovered = tracker.update(*_sample(4, [[52, 20, 60, 40]]), timestamp_s=12.1)
    assert recovered.track_ids.tolist() == first.track_ids.tolist()
    assert tracker.active_tracks[0].first_timestamp_s == 10.0


def test_seconds_covariance_composes_across_subdivided_prediction_gaps() -> None:
    """The full timed covariance integrates noise without losing cross terms."""
    noise = KalmanNoiseConfig(time_unit="seconds", reference_dt_s=0.1)
    prior = noise.initial_covariance(INITIAL_COVARIANCE.copy(), 2)
    direct = predict_covariance(prior, dt=0.7, noise_config=noise)
    split = predict_covariance(prior, dt=0.2, noise_config=noise)
    split = predict_covariance(split, dt=0.5, noise_config=noise)
    np.testing.assert_allclose(split, direct, rtol=1e-12, atol=1e-12)
    assert direct[0, 2] > 0.0
    assert np.linalg.eigvalsh(direct).min() > 0.0


@pytest.mark.parametrize("per_class", [False, True])
def test_gaussian_noise_profiles_use_canonical_class_ids(per_class: bool) -> None:
    """State and measurement settings follow original class IDs through kernel encoding."""
    special_class = 2**55 + 1
    noise = KalmanNoiseConfig(by_class={special_class: KalmanNoiseConfig(initial_velocity_scale=4.0)})
    tracker = MafHda(
        MafHdaConfig(s2ta_mode="motion", t2ta_mode="motion"),
        kalman=KalmanConfig(variable_dt=True, noise=noise),
        per_class=per_class,
    )
    detections, image = _sample(0, [[10, 20, 20, 40], [40, 20, 50, 40]], classes=[special_class, 2])
    result = tracker.update(detections, image, timestamp_s=4.0)
    by_id = {track.id: track.covariance[2, 2] for track in tracker.active_tracks}
    variances = {cls: by_id[track_id] for cls, track_id in zip(result.class_ids.tolist(), result.track_ids.tolist())}
    assert variances[special_class] == pytest.approx(4.0 * variances[2])
