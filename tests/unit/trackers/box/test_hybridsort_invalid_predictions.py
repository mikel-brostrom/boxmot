"""Invalid HybridSORT predictions must not shift the surviving track indices."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from boxmot import HybridSortConfig
from boxmot.structures import Boxes, Detections
from boxmot.trackers.hybridsort.tracker import HybridSort


def _detections(rows: np.ndarray, embeddings: np.ndarray | None) -> Detections:
    """Construct a batch with optional precomputed appearance features."""
    return Detections(
        geometry=Boxes(torch.as_tensor(rows[:, :4], dtype=torch.float32)),
        scores=torch.as_tensor(rows[:, 4], dtype=torch.float32),
        class_ids=torch.zeros(len(rows), dtype=torch.int64),
        sample_id="sequence/frame",
        embeddings=None if embeddings is None else torch.as_tensor(embeddings, dtype=torch.float32),
    )


@pytest.mark.parametrize("use_embeddings", [False, True])
@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("invalid_component", ["box", "kalman_confidence", "simple_confidence"])
def test_invalid_prediction_retains_the_surviving_identity(
    monkeypatch: pytest.MonkeyPatch,
    use_embeddings: bool,
    invalid_value: float,
    invalid_component: str,
) -> None:
    """Dropping prediction zero must also drop track zero before association."""
    tracker = HybridSort(
        config=HybridSortConfig(
            cmc_method=None,
            use_embeddings=use_embeddings,
            use_byte=False,
            min_hits=1,
            det_thresh=0.5,
            iou_threshold=0.3,
            asso_func="iou",
            max_age=20,
            max_obs=30,
        )
    )
    rows = np.array([[10, 10, 30, 40, 0.95], [100, 100, 120, 140, 0.95]], dtype=float)
    embeddings = np.eye(2) if use_embeddings else None
    first = tracker.update(_detections(rows, embeddings))
    identities = dict(zip(first.detection_indices.tolist(), first.track_ids.tolist(), strict=True))
    invalid_track = tracker.active_tracks[0]
    finish_prediction = invalid_track._finish_prediction

    def nonfinite_prediction() -> tuple[np.ndarray, float, float]:
        box, kalman_confidence, simple_confidence = finish_prediction()
        if invalid_component == "box":
            box = box.copy()
            box[0, 0] = invalid_value
        elif invalid_component == "kalman_confidence":
            kalman_confidence = invalid_value
        else:
            simple_confidence = invalid_value
        return box, kalman_confidence, simple_confidence

    monkeypatch.setattr(invalid_track, "_finish_prediction", nonfinite_prediction)
    second = tracker.update(_detections(rows[1:], None if embeddings is None else embeddings[1:]))

    assert second.track_ids.tolist() == [identities[1]]
    assert len(tracker.active_tracks) == 1
    assert tracker.active_tracks[0].id == identities[1]
    assert torch.isfinite(second.geometry.values).all()
