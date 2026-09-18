"""Exercise mask cues through cost-based trackers without loading EdgeTAM."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
import torch

from boxmot import BotSortConfig, SFSORTConfig, StrongSortConfig
from boxmot.structures import Boxes, Detections, Tracks
from boxmot.trackers import MaskGuidance, MaskGuidanceConfig
from boxmot.trackers.botsort.tracker import BotSort
from boxmot.trackers.sfsort.tracker import SFSORT
from boxmot.trackers.strongsort.tracker import StrongSort

FRAME = np.zeros((96, 128, 3), dtype=np.uint8)


@dataclass
class _Propagator:
    """Return controlled masks through the real guidance lifecycle."""

    device: str = "cpu"
    max_objects: int = 96
    prompt_overlap: float = 0.10
    masks: dict[int, np.ndarray] = field(default_factory=dict)

    def propagate(self, frame_index, frame, active_boxes, new_boxes) -> dict[int, np.ndarray]:
        return self.masks.copy()

    def retain_tracks(self, track_ids: set[int]) -> None:
        self.masks = {identity: mask for identity, mask in self.masks.items() if identity in track_ids}

    def reset(self) -> None:
        self.masks.clear()


def _tracker(name: str, *, guided: bool = False, **kwargs):
    """Disable unrelated image registration and supply embeddings in updates."""
    propagator = _Propagator()
    guidance = MaskGuidance(MaskGuidanceConfig("edgetam.pt", device="cpu"), propagator=propagator) if guided else None
    if name == "botsort":
        tracker = BotSort(
            config=BotSortConfig(
                use_cmc=False,
                track_low_thresh=0.1,
                track_high_thresh=0.5,
                second_match_thresh=kwargs.pop("second_match_thresh", 0.5),
                use_embeddings=kwargs.pop("use_embeddings", False),
                **kwargs,
            ),
            mask_guidance=guidance,
        )
    elif name == "sfsort":
        tracker = SFSORT(config=SFSORTConfig(match_th_second=0.5, **kwargs), mask_guidance=guidance)
    else:
        tracker = StrongSort(config=StrongSortConfig(n_init=1, **kwargs), mask_guidance=guidance)
        tracker.cmc = None
    return tracker, propagator


def _rows(score: float = 0.95) -> np.ndarray:
    return np.array([[10, 10, 30, 40, score, 0], [14, 10, 34, 40, score, 0]], dtype=np.float32)


def _update(tracker, rows: np.ndarray) -> np.ndarray:
    embeddings = np.tile([1.0, 0.0], (len(rows), 1)).astype(np.float32)
    if isinstance(tracker, BotSort) and tracker.use_embeddings and len(rows) > 1:
        embeddings[1] = [0.8, 0.6]
    needs_embeddings = isinstance(tracker, StrongSort) or isinstance(tracker, BotSort) and tracker.use_embeddings
    inputs = (
        Detections(
            geometry=Boxes(torch.from_numpy(np.ascontiguousarray(rows[:, :4]))),
            scores=torch.from_numpy(np.ascontiguousarray(rows[:, 4])),
            class_ids=torch.from_numpy(np.ascontiguousarray(rows[:, 5], dtype=np.int64)),
            embeddings=torch.from_numpy(embeddings),
            sample_id=f"test/{tracker.frame_count}",
        )
        if needs_embeddings
        else rows
    )
    result = tracker.update(inputs, FRAME)
    return result.to_aabb_rows().numpy() if isinstance(result, Tracks) else result


def _mask(box: np.ndarray) -> np.ndarray:
    foreground = np.zeros(FRAME.shape[:2], dtype=bool)
    x1, y1, x2, y2 = box.astype(int)
    foreground[y1:y2, x1:x2] = True
    return foreground


@pytest.mark.parametrize(
    ("name", "stage"),
    [
        ("botsort", "high"),
        ("botsort", "low"),
        ("botsort", "appearance"),
        ("sfsort", "high"),
        ("sfsort", "low"),
        ("strongsort", "appearance"),
        ("strongsort", "fallback"),
    ],
)
def test_temporal_masks_change_ambiguous_assignments(name: str, stage: str, monkeypatch: pytest.MonkeyPatch) -> None:
    options = {"use_embeddings": True} if name == "botsort" and stage == "appearance" else {}
    baseline, _ = _tracker(name, **options)
    guided, propagator = _tracker(name, guided=True, **options)
    original = _rows()
    identities = _update(guided, original)[:, 4].astype(int)
    _update(baseline, original)
    assert len(identities) == 2
    for identity, box in zip(identities, original[::-1, :4]):
        propagator.masks[identity] = _mask(box)
    if stage == "fallback":
        # The appearance stage cannot admit these pairs even with a mask bonus;
        # both trackers therefore use their normal IoU fallback stage.
        for tracker in (baseline, guided):
            monkeypatch.setattr(
                tracker.metric, "distance", lambda features, targets: np.full((len(targets), len(features)), 2.0)
            )
    rows = _rows(0.3 if stage == "low" else 0.95)

    expected = _update(baseline, rows)
    actual = _update(guided, rows)

    np.testing.assert_array_equal(expected[:, [4, 7]], np.column_stack((identities, [0, 1])))
    np.testing.assert_array_equal(actual[:, [4, 7]], np.column_stack((identities, [1, 0])))
    assert actual.dtype == expected.dtype
    assert actual.flags.c_contiguous


@pytest.mark.parametrize("name", ["botsort", "sfsort", "strongsort"])
@pytest.mark.parametrize("threshold", [0.1, 0.8])
def test_absent_masks_preserve_tracking_with_configured_thresholds(name: str, threshold: float) -> None:
    options = {
        "botsort": {"match_thresh": threshold, "second_match_thresh": threshold},
        "sfsort": {"match_th_first": threshold},
        "strongsort": {"max_cos_dist": threshold, "max_iou_dist": threshold},
    }[name]
    baseline, _ = _tracker(name, **options)
    guided, _ = _tracker(name, guided=True, **options)
    moved = _rows().copy()
    moved[:, [0, 2]] += 15
    for rows in (_rows(), moved, _rows(0.3), np.empty((0, 6), dtype=np.float32), _rows()):
        np.testing.assert_array_equal(_update(guided, rows), _update(baseline, rows))


@pytest.mark.parametrize("name", ["botsort", "sfsort"])
def test_masks_do_not_recover_geometrically_isolated_identity(name: str) -> None:
    baseline, _ = _tracker(name)
    guided, propagator = _tracker(name, guided=True)
    original = _rows()[:1]
    identity = int(_update(guided, original)[0, 4])
    _update(baseline, original)
    moved = original.copy()
    moved[:, [0, 2]] += 60
    propagator.masks[identity] = _mask(moved[0, :4])

    expected = _update(baseline, moved)
    actual = _update(guided, moved)

    assert identity not in expected[:, 4]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("mc_lambda", [0.0, 1e-5, 0.98])
def test_strongsort_preserves_motion_gate_even_when_mask_bonus_would_admit_pair(
    mc_lambda: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracker, _ = _tracker("strongsort", guided=True, mc_lambda=mc_lambda, max_cos_dist=9.0)
    identity = int(_update(tracker, _rows()[:1])[0, 4])
    track = tracker.tracks[0]
    track.time_since_update = 2  # Lost candidates do not enter the one-frame IoU fallback.
    tracker._mask_guidance._masks = {identity: _mask(_rows()[0, :4])}
    monkeypatch.setattr(track.kf, "gating_distance", lambda *args: np.array([9.6]))
    detection = type("Detection", (), {})()
    detection.feat = np.array([1.0, 0.0])
    detection.xyxy = _rows()[0, :4]
    detection.to_measurement = lambda: track.bbox.copy()

    matches, unmatched_tracks, unmatched_detections = tracker._match([detection])

    assert matches == []
    assert unmatched_tracks == [0]
    assert unmatched_detections == [0]
