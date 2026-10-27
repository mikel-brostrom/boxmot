"""BoT-SORT observations preserve geometry, appearance, and track lifecycles."""

import numpy as np
import pytest
import torch

from boxmot import BotSort, BotSortConfig
from boxmot.structures import Boxes, Detections, OrientedBoxes
from boxmot.trackers.botsort.track import STrack, _Detection
from boxmot.trackers.common.motion.kalman_filters.config import KalmanConfig

KINDS = ("xyah", "xywh", "xysr", "xyhr", "xyscr")


class _FullObservationBotSort(BotSort):
    """Reference the original full-track detection allocation and birth path."""

    def _create_detections(self, rows: np.ndarray, features: np.ndarray | None = None) -> list[STrack]:
        return [
            STrack(
                row,
                features[index] if self.use_embeddings and features is not None else None,
                id_allocator=self.id_allocator,
                max_obs=self.max_obs,
                alpha=self.config.alpha_fixed_emb,
                is_obb=self.is_obb,
                motion_model=self.motion_model,
            )
            for index, row in enumerate(rows)
        ]

    def _initialize_new_tracks(self, unmatched, activated, detections) -> None:
        for index in unmatched:
            track = detections[index]
            if track.conf >= self.new_track_thresh:
                track.activate(self.kalman_filter, self.frame_count)
                activated.append(track)


def _rows(is_obb: bool, frame: int, observations: tuple[tuple[int, float], ...]) -> np.ndarray:
    """Move boxes through equivalent OBB forms and occasional class changes."""
    rows = []
    for identity, confidence in observations:
        x, y = 25.123456789 + identity * 48 + frame * 0.1337, 45.345678912
        width, height = 18.123456789, 32.345678912
        if is_obb:
            angle = 0.234567891 + frame * 0.001
            if frame % 3 == 2:
                width, height, angle = height, width, angle + np.pi / 2
            box = [x, y, width, height, angle]
        else:
            box = [x, y, x + width, y + height]
        rows.append([*box, confidence, (identity + frame // 3) % 2])
    return np.asarray(rows, dtype=np.float64).reshape(-1, 7 if is_obb else 6)


@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_observation_geometry_features_and_birth_match_full_tracks(is_obb: bool, dtype) -> None:
    """Prepared geometry and normalized features retain exact scalar rounding."""
    options = dict(config=BotSortConfig(use_cmc=False), is_obb=is_obb)
    tracker, reference = BotSort(**options), _FullObservationBotSort(**options)
    rows = _rows(is_obb, 2, ((0, 0.9123456789), (1, 0.9234567891))).astype(dtype)
    indexed = np.column_stack((rows, np.arange(len(rows), dtype=dtype)))
    features = np.array([[0.123456789, 0.987654321, 0.234567891], [0.789123456, 0.432198765, 0.567891234]], dtype=dtype)
    records, expected = tracker._create_detections(indexed, features), reference._create_detections(indexed, features)
    for record, full in zip(records, expected, strict=True):
        np.testing.assert_array_equal(record.xywh, full.xywh)
        np.testing.assert_array_equal(record.xyxy, full.xyxy)
        np.testing.assert_array_equal(record.xywha, full.xywha)
        np.testing.assert_array_equal(record.curr_feat, full.curr_feat)
        assert not hasattr(record, "history_observations")
        assert not hasattr(record, "features")
        assert not np.shares_memory(record.raw_row, indexed)
        assert not np.shares_memory(record.curr_feat, features)
        born = STrack(record, id_allocator=tracker.id_allocator, is_obb=is_obb, motion_model=tracker.motion_model)
        np.testing.assert_array_equal(born.curr_feat, full.curr_feat)
        np.testing.assert_array_equal(born.smooth_feat, full.smooth_feat)
        np.testing.assert_array_equal(born.features, full.features)
        assert born.cls_hist == full.cls_hist
    indexed[:] = 0
    features[:] = 0
    geometry = _Detection.batch_geometry(records, is_obb=is_obb)
    np.testing.assert_array_equal(geometry, np.asarray([track.xywha if is_obb else track.xyxy for track in expected]))
    geometry[:] = 0
    np.testing.assert_array_equal(
        _Detection.batch_geometry(records, is_obb=is_obb),
        np.asarray([track.xywha if is_obb else track.xyxy for track in expected]),
    )


@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("embeddings", [False, True])
@pytest.mark.parametrize("motion_fusion", ["off", "blend"])
def test_observations_preserve_all_matching_stages_and_lifecycles(is_obb, kind, embeddings, motion_fusion) -> None:
    """Compare complete trajectories, including tentative tracks, gaps and expiry."""
    options = dict(
        config=BotSortConfig(use_cmc=False, use_embeddings=embeddings, max_lost_frames=2, motion_fusion=motion_fusion),
        kalman=KalmanConfig(type=kind),
        is_obb=is_obb,
    )
    actual, expected = BotSort(**options), _FullObservationBotSort(**options)
    observations = (
        (), ((0, 0.95), (1, 0.91)), ((0, 0.96), (1, 0.93), (2, 0.98)),
        ((0, 0.35), (1, 0.96)), (), ((0, 0.97), (1, 0.94)), ((0, 0.95),),
        (), (), (), (), ((3, 0.96),), ((3, 0.98),),
    )
    for frame, visible in enumerate(observations):
        rows = _rows(is_obb, frame, visible)
        if embeddings:
            features = np.array([[1 + identity, 0.234567891 + frame * 0.03, 0.345678912] for identity, _ in visible])
            data = Detections(
                geometry=(OrientedBoxes if is_obb else Boxes)(torch.tensor(rows[:, :-2], dtype=torch.float32)),
                scores=torch.tensor(rows[:, -2], dtype=torch.float32),
                class_ids=torch.tensor(rows[:, -1], dtype=torch.int64),
                embeddings=torch.tensor(features.reshape(-1, 3), dtype=torch.float32),
                sample_id=f"sequence/{frame}",
            )
            outputs, reference = actual.update(data), expected.update(data)
            serialize = "to_obb_rows" if is_obb else "to_aabb_rows"
            np.testing.assert_array_equal(getattr(outputs, serialize)(), getattr(reference, serialize)())
        else:
            np.testing.assert_array_equal(actual.update(rows), expected.update(rows))
        for name in ("active_tracks", "lost_stracks", "removed_stracks"):
            tracks, references = getattr(actual, name), getattr(expected, name)
            assert len(tracks) == len(references)
            for track, reference in zip(tracks, references, strict=True):
                assert type(track) is STrack
                assert track.meta == reference.meta
                assert track.cls_hist == reference.cls_hist
                assert (track.state, track.tracklet_len) == (reference.state, reference.tracklet_len)
                for attr in ("mean", "covariance", "history_observations", "curr_feat", "smooth_feat", "features"):
                    np.testing.assert_array_equal(getattr(track, attr), getattr(reference, attr))


def test_only_new_identities_allocate_persistent_track_state(monkeypatch) -> None:
    """Matched and rejected low-score detections need no identity histories."""
    created = []
    constructor = STrack.__init__

    def record_construction(self, *args, **kwargs) -> None:
        constructor(self, *args, **kwargs)
        created.append(self)

    monkeypatch.setattr(STrack, "__init__", record_construction)
    tracker = BotSort(BotSortConfig(use_cmc=False, use_embeddings=False))
    tracker.update(_rows(False, 0, ((0, 0.95), (1, 0.95))))
    assert len(created) == 2
    tracker.update(_rows(False, 1, ((0, 0.95), (1, 0.35), (2, 0.35))))
    assert len(created) == 2
    tracker.update(_rows(False, 2, ((0, 0.95), (1, 0.95), (2, 0.95))))
    assert len(created) == 3
