"""OccluBoost expires identities at the configured track lifetime."""

from __future__ import annotations

import pytest
import torch

from boxmot.structures import Boxes, Detections, OrientedBoxes
from boxmot.trackers.occluboost.tracker import OccluBoost


@pytest.mark.parametrize("is_obb", [False, True])
def test_expired_track_gets_a_new_identity_despite_identical_appearance(is_obb: bool) -> None:
    """An object returning after expiration cannot recover the removed identity."""
    tracker = OccluBoost(
        is_obb=is_obb,
        use_embeddings=True,
        use_cmc=False,
        use_dlo_boost=False,
        use_duo_boost=False,
        min_hits=1,
        max_age=2,
        obb_max_age=2,
    )
    geometry_type = OrientedBoxes if is_obb else Boxes
    geometry = torch.tensor([[30, 40, 20, 40, 0.2]] if is_obb else [[20, 20, 40, 60]], dtype=torch.float32)

    def observation(frame: int, *, present: bool = True) -> Detections:
        """Keep geometry and appearance fixed across the disappearance."""
        count = int(present)
        return Detections(
            geometry_type(geometry[:count]),
            torch.full((count,), 0.95),
            torch.zeros(count, dtype=torch.int64),
            sample_id=f"lifecycle/{frame}",
            embeddings=torch.tensor([[1, 0, 0]], dtype=torch.float32)[:count],
        )

    first = tracker.update(observation(0))
    assert len(first) == 1
    identity = first.track_ids.item()
    for frame in range(1, 6):
        confirmed = tracker.update(observation(frame))
        assert confirmed.track_ids.tolist() == [identity]

    for missed in range(1, 4):
        missing = tracker.update(observation(5 + missed, present=False))
        assert len(missing) == 0
        assert len(tracker.trackers) == int(missed <= 2)

    newborn = tracker.update(observation(9))
    assert len(newborn) == 0
    assert len(tracker.trackers) == 1
    assert tracker.trackers[0].id != identity

    returned = tracker.update(observation(10))

    assert len(returned) == 1
    assert returned.track_ids.item() != identity
    torch.testing.assert_close(returned.geometry.values, geometry)
