"""StrongSORT public updates preserve track and gallery lifecycle ownership."""

import numpy as np
import pytest
import torch

from boxmot import StrongSort, StrongSortConfig
from boxmot.structures import Boxes, Detections, OrientedBoxes, Tracks


def _update(tracker: StrongSort, class_ids: list[int]) -> Tracks:
    """Supply deterministic observations, including a valid empty embedding matrix."""
    box = [20.0, 20.0, 20.0, 20.0, 0.2] if tracker.is_obb else [10.0, 10.0, 30.0, 30.0]
    boxes = np.repeat(np.asarray(box, dtype=np.float32)[None], len(class_ids), axis=0)
    geometry = OrientedBoxes if tracker.is_obb else Boxes
    embeddings = [[1.0, 0.0] if class_id in (0, 2) else [0.0, 1.0] for class_id in class_ids]
    return tracker.update(
        Detections(
            geometry=geometry(torch.from_numpy(boxes)),
            scores=torch.full((len(class_ids),), 0.9),
            class_ids=torch.tensor(class_ids, dtype=torch.int64),
            embeddings=torch.tensor(embeddings, dtype=torch.float32).reshape(-1, 2),
        )
    )


@pytest.mark.parametrize("is_obb", [False, True])
def test_active_pool_retains_confirmed_tracks_until_expiry(is_obb: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """Confirmation, missed frames, and expiry operate on the publicly visible pool."""
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    tracker = StrongSort(StrongSortConfig(use_cmc=False, n_init=2, max_age=2), is_obb=is_obb)

    assert len(_update(tracker, [0])) == 0
    track = tracker.active_tracks[0]
    assert track.is_tentative()
    assert not tracker.metric.samples

    confirmed = _update(tracker, [0])
    assert confirmed.track_ids.tolist() == [track.id]
    assert tracker.active_tracks == [track]
    assert track.is_confirmed()
    assert track.hits == 2

    for missed_frames in (1, 2):
        assert len(_update(tracker, [])) == 0
        assert tracker.active_tracks == [track]
        assert track.time_since_update == missed_frames
        assert set(tracker.metric.samples) == {track.id}

    assert len(_update(tracker, [])) == 0
    assert track.is_deleted()
    assert tracker.active_tracks == []
    assert tracker.metric.samples == {}

    assert len(_update(tracker, [0])) == 0
    replacement = _update(tracker, [0])
    assert replacement.track_ids.item() != track.id


@pytest.mark.parametrize("is_obb", [False, True])
def test_unconfirmed_track_is_removed_after_one_miss(is_obb: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """A tentative identity cannot survive a gap or enter the appearance gallery."""
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    tracker = StrongSort(StrongSortConfig(use_cmc=False, n_init=2), is_obb=is_obb)
    _update(tracker, [0])
    track = tracker.active_tracks[0]

    assert len(_update(tracker, [])) == 0
    assert track.is_deleted()
    assert tracker.active_tracks == []
    assert tracker.metric.samples == {}


@pytest.mark.parametrize("is_obb", [False, True])
def test_per_class_tracks_and_galleries_age_independently(is_obb: bool) -> None:
    """Updating one class preserves another's gallery until that identity expires."""
    tracker = StrongSort(
        StrongSortConfig(use_cmc=False, n_init=1, max_age=1),
        per_class=True,
        class_ids=[2, 5],
        is_obb=is_obb,
    )
    initial = _update(tracker, [5, 2])
    ids = dict(zip(initial.class_ids.tolist(), initial.track_ids.tolist()))
    tracks = {class_id: tracker.get_class_tracks(class_id)[0] for class_id in (2, 5)}
    galleries = {class_id: tracker.get_class_track_state(class_id).attrs["strongsort_metric"] for class_id in (2, 5)}
    assert galleries[2] is not galleries[5]
    assert [track.cls for track in tracker.active_tracks] == [2, 5]
    assert initial.detection_indices.tolist() == [1, 0]
    for class_id in (2, 5):
        assert set(galleries[class_id].samples) == {ids[class_id]}

    visible = _update(tracker, [5])
    assert visible.track_ids.tolist() == [ids[5]]
    assert tracker.active_tracks == [tracks[2], tracks[5]]
    assert tracks[2].time_since_update == 1
    assert set(galleries[2].samples) == {ids[2]}
    assert set(galleries[5].samples) == {ids[5]}

    _update(tracker, [5])
    assert tracker.active_tracks == [tracks[5]]
    assert tracker.get_class_tracks(2) == []
    assert galleries[2].samples == {}
    assert set(galleries[5].samples) == {ids[5]}

    returned = _update(tracker, [5, 2])
    current_ids = dict(zip(returned.class_ids.tolist(), returned.track_ids.tolist()))
    assert current_ids[5] == ids[5]
    assert current_ids[2] != ids[2]
    assert returned.detection_indices.tolist() == [1, 0]


@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("per_class", [False, True])
def test_reset_clears_tracks_and_galleries_and_restarts_identity_allocation(is_obb: bool, per_class: bool) -> None:
    """A reset starts an independent sequence with no retained tracks or embeddings."""
    tracker = StrongSort(
        StrongSortConfig(use_cmc=False, n_init=1), per_class=per_class, class_ids=[2, 5], is_obb=is_obb
    )
    initial = _update(tracker, [5, 2])
    previous_tracks = list(tracker.active_tracks)
    _update(tracker, [5])

    tracker.reset()

    assert tracker.active_tracks == []
    assert tracker.metric.samples == {}
    assert not tracker.class_track_states
    assert tracker.frame_count == 0
    restarted = _update(tracker, [5, 2])
    assert restarted.track_ids.tolist() == initial.track_ids.tolist()
    assert restarted.class_ids.tolist() == initial.class_ids.tolist()
    assert all(track not in previous_tracks for track in tracker.active_tracks)
