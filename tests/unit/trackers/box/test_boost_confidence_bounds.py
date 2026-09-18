"""Large searchable DLO coefficients preserve the public confidence contract."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from boxmot import BoostTrackConfig, OccluBoostConfig
from boxmot.structures import Boxes, Detections, OrientedBoxes, Tracks
from boxmot.trackers.boosttrack.tracker import BoostTrack
from boxmot.trackers.occluboost.tracker import OccluBoost


@pytest.mark.parametrize("tracker_type", [BoostTrack, OccluBoost])
@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("canonical", [False, True])
@pytest.mark.parametrize("rich_similarity", [False, True])
@pytest.mark.parametrize("coefficient", [0.8, 1.5404905430267744])
def test_dlo_boost_keeps_valid_scores_and_inputs(
    tracker_type: type[BoostTrack],
    is_obb: bool,
    canonical: bool,
    rich_similarity: bool,
    coefficient: float,
) -> None:
    """Real updates retain identity and saturate only an out-of-range score."""
    config_type = OccluBoostConfig if tracker_type is OccluBoost else BoostTrackConfig
    tracker = tracker_type(
        is_obb=is_obb,
        config=config_type(
            use_embeddings=False,
            use_cmc=False,
            use_dlo_boost=True,
            use_duo_boost=False,
            use_rich_s=rich_similarity,
            use_sb=False,
            use_vt=False,
            dlo_boost_coef=coefficient,
            min_hits=1,
        ),
    )
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    coordinates = [25, 45, 30, 70, 0.0] if is_obb else [10, 10, 40, 80]
    identities = []
    final_score = None
    for frame_index, confidence in enumerate([1.0, 0.6, 0.6]):
        rows = np.asarray([[*coordinates, confidence, 0]], dtype=np.float32)
        original = rows.copy()
        if canonical:
            geometry_type = OrientedBoxes if is_obb else Boxes
            detections = Detections(
                geometry=geometry_type(torch.from_numpy(rows[:, :-2].copy())),
                scores=torch.from_numpy(rows[:, -2].copy()),
                class_ids=torch.zeros(1, dtype=torch.int64),
                sample_id=f"frame-{frame_index}",
            )
            result = tracker.update(detections, image)
            assert isinstance(result, Tracks)
            np.testing.assert_array_equal(detections.geometry.values.numpy(), original[:, :-2])
            np.testing.assert_array_equal(detections.scores.numpy(), original[:, -2])
            identities.append(result.track_ids.tolist())
            scores = result.scores.numpy()
        else:
            result = tracker.update(rows, image)
            assert result.shape == (1, 9 if is_obb else 8)
            identities.append(result[:, -4].tolist())
            scores = result[:, -3]
        np.testing.assert_array_equal(rows, original)
        assert len(scores) == 1
        assert np.all((scores >= confidence) & (scores <= 1.0))
        final_score = float(scores[0])
    assert identities[0] == identities[1] == identities[2]
    if coefficient > 1:
        assert final_score == 1.0
    elif not rich_similarity:
        assert final_score == pytest.approx(coefficient)
