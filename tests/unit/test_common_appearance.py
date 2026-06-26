import numpy as np

from boxmot.trackers.common.appearance import (
    blend_embeddings,
    confidence_aware_alpha,
    ema_update_embedding,
    normalize_embedding,
    resolve_batch_embeddings,
)
from boxmot.trackers.common.detections import DetectionBatch
from boxmot.trackers.common.detections.layout import AABB_DETECTIONS
from boxmot.trackers.common.tracking.track import TrackIdAllocator
from boxmot.trackers.common.track_models.botsort import STrack


class DummyReID:
    def __init__(self) -> None:
        self.calls = 0

    def get_features(self, boxes: np.ndarray, img: np.ndarray) -> np.ndarray:
        self.calls += 1
        return np.full((len(boxes), 3), 2.0, dtype=np.float32)


def _batch(embs: np.ndarray | None = None) -> DetectionBatch:
    dets = np.array([[1, 2, 11, 22, 0.9, 3]], dtype=np.float32)
    return DetectionBatch.from_layout(dets, AABB_DETECTIONS, embs=embs)


def test_normalize_embedding_returns_unit_copy_without_mutating_input():
    feature = np.array([3.0, 4.0], dtype=np.float32)
    original = feature.copy()

    normalized = normalize_embedding(feature)

    np.testing.assert_allclose(feature, original)
    np.testing.assert_allclose(normalized, np.array([0.6, 0.8], dtype=np.float32))
    assert normalized is not feature


def test_normalize_embedding_handles_zero_vector():
    normalized = normalize_embedding(np.zeros(4, dtype=np.float32))

    np.testing.assert_allclose(normalized, np.zeros(4, dtype=np.float32))
    assert np.isfinite(normalized).all()


def test_ema_update_embedding_matches_existing_tracker_formula():
    previous = np.array([2.0, 0.0], dtype=np.float32)
    current = np.array([0.0, 4.0], dtype=np.float32)
    alpha = 0.75
    expected = (alpha * previous) + ((1.0 - alpha) * current)
    expected = expected / np.linalg.norm(expected)

    updated = ema_update_embedding(previous, current, alpha=alpha)

    np.testing.assert_allclose(updated, expected)


def test_blend_embeddings_matches_weighted_normalized_blend():
    previous = np.array([1.0, 0.0], dtype=np.float32)
    current = np.array([0.0, 1.0], dtype=np.float32)
    expected = np.array([0.2, 0.8], dtype=np.float32)
    expected = expected / np.linalg.norm(expected)

    blended = blend_embeddings(previous, current, 0.2, 0.8)

    np.testing.assert_allclose(blended, expected)


def test_resolve_batch_embeddings_reuses_precomputed_embeddings():
    embs = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    model = DummyReID()

    resolved = resolve_batch_embeddings(_batch(embs), np.zeros((8, 8, 3), dtype=np.uint8), model=model)

    np.testing.assert_allclose(resolved, embs)
    assert model.calls == 0


def test_resolve_batch_embeddings_returns_disabled_placeholder():
    model = DummyReID()

    resolved = resolve_batch_embeddings(
        _batch(),
        np.zeros((8, 8, 3), dtype=np.uint8),
        model=model,
        enabled=False,
        placeholder_value=0.0,
    )

    np.testing.assert_allclose(resolved, np.zeros((1, 1), dtype=np.float32))
    assert model.calls == 0


def test_resolve_batch_embeddings_skips_model_for_empty_batch():
    model = DummyReID()
    batch = _batch().select([])

    resolved = resolve_batch_embeddings(batch, np.zeros((8, 8, 3), dtype=np.uint8), model=model)

    assert resolved.shape == (0, 1)
    assert model.calls == 0


def test_resolve_batch_embeddings_extracts_features_for_nonempty_batch():
    model = DummyReID()

    resolved = resolve_batch_embeddings(_batch(), np.zeros((8, 8, 3), dtype=np.uint8), model=model)

    np.testing.assert_allclose(resolved, np.full((1, 3), 2.0, dtype=np.float32))
    assert model.calls == 1


def test_confidence_aware_alpha_matches_tracker_formula():
    confs = np.array([0.5, 0.75, 1.0], dtype=np.float32)
    det_thresh = 0.5
    base_alpha = 0.9
    trust = (confs - det_thresh) / (1.0 - det_thresh)
    expected = base_alpha + (1.0 - base_alpha) * (1.0 - trust)

    alpha = confidence_aware_alpha(confs, det_thresh, base_alpha=base_alpha)

    np.testing.assert_allclose(alpha, expected)


def test_confidence_aware_alpha_handles_empty_input():
    alpha = confidence_aware_alpha(np.empty(0, dtype=np.float32), 0.5)

    assert alpha.shape == (0,)


def test_botsort_feature_update_does_not_mutate_input_embeddings():
    det = np.array([10, 10, 30, 60, 0.95, 1, 0], dtype=np.float32)
    initial_feature = np.array([3.0, 4.0, 0.0, 0.0], dtype=np.float32)
    next_feature = np.array([0.0, 0.0, 5.0, 0.0], dtype=np.float32)
    initial_original = initial_feature.copy()
    next_original = next_feature.copy()

    track = STrack(det, feat=initial_feature, id_allocator=TrackIdAllocator())
    track.update_features(next_feature)

    np.testing.assert_allclose(initial_feature, initial_original)
    np.testing.assert_allclose(next_feature, next_original)
    np.testing.assert_allclose(np.linalg.norm(track.curr_feat), 1.0)
    np.testing.assert_allclose(np.linalg.norm(track.smooth_feat), 1.0)
