"""OccluBoost's association stages retain scalar motion and AMS behavior."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch

import boxmot.trackers.occluboost.tracker as occluboost_module
from boxmot.structures import Boxes, Detections, OrientedBoxes
from boxmot.trackers.boosttrack.track import KalmanBoxTracker
from boxmot.trackers.occluboost.tracker import OccluBoost


def _tracker(*, is_obb: bool, adaptive: bool = False) -> OccluBoost:
    """Use supplied embeddings and allow every relevant association stage."""
    return OccluBoost(
        is_obb=is_obb,
        use_embeddings=True,
        use_cmc=False,
        use_dlo_boost=False,
        use_duo_boost=False,
        det_thresh=0.5,
        obb_det_thresh=0.5,
        min_hits=1,
        new_track_thresh=0.5,
        instant_confirm_thresh=0.5,
        obb_new_track_thresh=0.5,
        obb_instant_confirm_thresh=0.5,
        iou_threshold=0.1,
        obb_iou_threshold=0.1,
        second_iou_thresh=0.0,
        obb_second_iou_thresh=0.0,
        second_pass_min_hits=0,
        recovery_iou_thresh=0.0,
        recovery_appearance_thresh=0.9,
        use_second_pass=True,
        adaptive_kf=adaptive,
    )


def _detections(frame: int, is_obb: bool) -> np.ndarray:
    """Move separated objects, partially occluding just one before restoring it."""
    centers = np.array([40, 130, 220], dtype=float) + frame * 0.35
    widths = np.array([20, 24, 28], dtype=float)
    heights = np.array([40, 44, 48], dtype=float)
    if 5 <= frame <= 7:
        widths[0] *= 0.4
        heights[0] *= 0.5
        centers[0] += 5
    if is_obb:
        return np.column_stack(
            (centers, np.full(3, 70), widths, heights, np.full(3, frame * 0.02), np.full(3, 0.95), np.zeros(3))
        )
    return np.column_stack(
        (centers - widths / 2, 70 - heights / 2, centers + widths / 2, 70 + heights / 2, np.full(3, 0.95), np.zeros(3))
    )


def _scalar_updates(tracker: OccluBoost, tracks: list[KalmanBoxTracker], detections: np.ndarray) -> None:
    """Run the previous per-track corrections and their original bookkeeping."""
    for track, det in zip(tracks, detections):
        if track.is_obb:
            tracker._ams_update_obb(track, det)
        else:
            tracker._ams_update(track, det)


@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("stage", ["first", "recovery", "second"])
def test_each_association_stage_batches_the_same_updates_as_scalar(
    is_obb: bool, stage: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real tracker runs preserve IDs, KF state, features, and observation buffers."""
    batched, scalar = (_tracker(is_obb=is_obb) for _ in range(2))
    image = np.zeros((160, 300, 3), dtype=np.uint8)
    embeddings = np.eye(3, dtype=np.float32)
    calls: list[int] = []
    original_update = batched._ams_multi_update

    def record_updates(tracks: list[KalmanBoxTracker], detections: np.ndarray) -> None:
        if tracks:
            calls.append(len(tracks))
        original_update(tracks, detections)

    monkeypatch.setattr(batched, "_ams_multi_update", record_updates)
    monkeypatch.setattr(scalar, "_ams_multi_update", lambda tracks, dets: _scalar_updates(scalar, tracks, dets))
    if not is_obb and stage == "recovery":

        def unmatched_first(detections: np.ndarray, tracks: np.ndarray, *args: object, **kwargs: object) -> tuple:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.arange(len(tracks)), None

        monkeypatch.setattr(occluboost_module, "associate", unmatched_first)
    for frame in range(14):
        detections = _detections(frame, is_obb)
        if frame and stage == "recovery":
            for tracker in (batched, scalar):
                # Tight geometry gate sends all observations to recovery.
                tracker.iou_threshold = 1.0
                tracker.obb_iou_threshold = 1.0
        if frame and stage == "second":
            detections[:, 5 if is_obb else 4] = 0.2
        dimensions = 5 if is_obb else 4
        geometry = OrientedBoxes if is_obb else Boxes
        observed = Detections(
            geometry(torch.from_numpy(detections[:, :dimensions]).float()),
            torch.from_numpy(detections[:, dimensions]).float(),
            torch.from_numpy(detections[:, dimensions + 1]).long(),
            sample_id=f"motion/{frame}",
            embeddings=torch.from_numpy(embeddings),
        )
        actual = batched.update(observed, image)
        expected = scalar.update(observed, image)
        torch.testing.assert_close(actual.geometry.values, expected.geometry.values, atol=2e-5, rtol=1e-6)
        for field in ("track_ids", "class_ids", "detection_indices", "scores"):
            torch.testing.assert_close(getattr(actual, field), getattr(expected, field), atol=0, rtol=0)
        assert len(batched.trackers) == len(scalar.trackers) == 3
        for track, reference in zip(batched.trackers, scalar.trackers):
            np.testing.assert_allclose(track.kf.x, reference.kf.x, atol=1e-10, rtol=1e-11)
            np.testing.assert_allclose(track.kf.P, reference.kf.P, atol=1e-10, rtol=1e-11)
            for field in ("id", "age", "time_since_update", "hit_streak", "det_ind", "is_activated"):
                assert getattr(track, field) == getattr(reference, field)
            np.testing.assert_array_equal(track.emb, reference.emb)
            np.testing.assert_allclose(track.history_observations, reference.history_observations, atol=2e-5)
            np.testing.assert_array_equal(getattr(track, "_ams_obs_buf", []), getattr(reference, "_ams_obs_buf", []))
    assert calls == [3] * 13


@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("adaptive", [False, True])
def test_batch_ams_retains_distinct_suppression_and_adaptive_noise(is_obb: bool, adaptive: bool) -> None:
    """One partially occluded track cannot change its neighbors' gain/noise policy."""
    coordinator = _tracker(is_obb=is_obb, adaptive=adaptive)
    detections = _detections(0, is_obb)
    indexed = np.column_stack((detections, np.arange(len(detections))))
    tracks = [
        KalmanBoxTracker(det, max_obs=30, is_obb=is_obb, adaptive_kf=adaptive, track_id=index)
        for index, det in enumerate(indexed)
    ]
    scalar = deepcopy(tracks)
    suppressed = False
    for frame in range(1, 35):
        indexed = np.column_stack((_detections(frame, is_obb), np.arange(len(detections))))
        for track, reference in zip(tracks, scalar):
            track.predict()
            reference.predict()
        if not is_obb:
            probes = deepcopy(tracks)
            alphas = [coordinator._compute_ams_alpha(track, det[:4]) for track, det in zip(probes, indexed)]
            suppressed |= min(alphas) < max(alphas)
        coordinator._ams_multi_update(tracks, indexed)
        _scalar_updates(coordinator, scalar, indexed)
        for track, reference in zip(tracks, scalar):
            np.testing.assert_allclose(track.kf.x, reference.kf.x, atol=1e-9, rtol=1e-10)
            np.testing.assert_allclose(track.kf.P, reference.kf.P, atol=1e-9, rtol=1e-10)
            np.testing.assert_allclose(track.kf.cov_update_policy.get_q(), reference.kf.cov_update_policy.get_q())
            np.testing.assert_allclose(track.kf.cov_update_policy.get_r(), reference.kf.cov_update_policy.get_r())
    if not is_obb:
        assert suppressed
