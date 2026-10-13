"""MAF-HDA confidence fusion respects logarithmic affinity costs and hard gates."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from boxmot import MafHda, MafHdaConfig
from boxmot.trackers.common.association.matching import linear_assignment
from boxmot.trackers.maf_hda.association import MAX_COST
from boxmot.trackers.maf_hda.tracker import _Observation


def _observation(confidence: float, *, offset: float = 0.0) -> _Observation:
    """Use one deterministic mask and box without image-dependent appearance."""
    return _Observation(
        bbox=np.array([10 + offset, 10, 30 + offset, 30], dtype=float),
        mask=np.ones((50, 100), dtype=bool),
        conf=confidence,
        cls=0,
        det_ind=0,
        weight=1.0,
    )


@pytest.mark.parametrize("fuse", [False, True])
@pytest.mark.parametrize("recovery", [False, True])
def test_score_fusion_changes_segment_assignment_only(fuse: bool, recovery: bool) -> None:
    """A confident nearby observation can outrank a less confident exact overlap."""
    tracker = MafHda(MafHdaConfig(s2ta_mode="motion", t2ta_mode="motion", fuse_detection_score=fuse))
    track = tracker._new_track(_observation(1.0), None)
    observations = [_observation(0.1), _observation(1.0, offset=4.0)]
    tracker.frame_count = 1
    candidates = [tracker._new_track(observation, None) for observation in observations] if recovery else None
    matches, _, _ = tracker._associate([track], observations, None, candidates=candidates)
    np.testing.assert_array_equal(matches, [[0, 1 if fuse and not recovery else 0]])


@pytest.mark.parametrize("centroid", [False, True])
def test_log_confidence_penalty_preserves_invalid_edges_and_cost_scale(
    centroid: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Confidence only increases valid costs; zero or invalid scores never open a gate."""
    tracker = MafHda(MafHdaConfig(
        s2ta_mode="motion", t2ta_mode="motion", asso_func="centroid" if centroid else "iou",
    ))
    tracker._initialize_frame_dimensions(width=100, height=50)
    track = tracker._new_track(_observation(1.0), None)
    observations = [
        _observation(1.0), _observation(0.5), _observation(0.0), _observation(np.nan),
        _observation(0.0, offset=80.0), _observation(0.01, offset=80.0),
    ]
    captured = []

    def capture(costs: np.ndarray, thresh: float | None):
        """Observe the real assignment input while retaining the normal solver."""
        captured.append(costs.copy())
        return linear_assignment(costs, thresh)

    monkeypatch.setattr("boxmot.trackers.maf_hda.tracker.linear_assignment", capture)
    tracker._associate([track], observations, None)
    tracker.config = replace(tracker.config, fuse_detection_score=True)
    tracker._associate([track], observations, None)
    original, fused = captured
    assert fused[0, 0] == original[0, 0]
    assert fused[0, 1] == pytest.approx(original[0, 1] - 100.0 * np.log(0.5))
    assert np.isinf(fused[0, 2:4]).all()
    assert np.all(fused >= original)
    np.testing.assert_array_equal(fused[0, 4:], original[0, 4:])
    assert np.all(np.isinf(fused[0, 4:])) if centroid else np.all(fused[0, 4:] == MAX_COST)
