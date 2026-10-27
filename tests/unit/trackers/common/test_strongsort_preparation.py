"""StrongSORT preparation keeps assignment order and appearance/motion costs."""

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from boxmot.trackers.common.association.matching import motion_gating_distance, motion_gating_threshold
from boxmot.trackers.common.association.strongsort import (
    INFTY_COST,
    NearestNeighborDistanceMetric,
    gate_cost_matrix,
    min_cost_matching,
)
from boxmot.trackers.common.motion.models import create_motion_model
from boxmot.trackers.strongsort.track import Track
from boxmot.trackers.strongsort.tracker import _Detection


@pytest.mark.parametrize("shape", [(2, 5), (5, 2), (4, 4), (1, 1), (0, 3), (3, 0)])
@pytest.mark.parametrize("tied", [False, True])
def test_unmatched_masks_preserve_original_partition_order(shape, tied: bool) -> None:
    """Rectangular, tied and rejected assignments keep caller index ordering."""
    costs = np.random.default_rng(734).uniform(size=shape)
    if tied:
        costs[:] = 0.5
    if costs.size:
        costs.flat[0] = 0.5  # An edge exactly at the threshold remains admissible.
    track_indices = [index * 3 + 2 for index in reversed(range(shape[0]))]
    detection_indices = [index * 2 + 1 for index in reversed(range(shape[1]))]
    threshold = 0.5
    actual = min_cost_matching(
        lambda *args: costs.copy(), threshold, [], [], track_indices, detection_indices
    )
    gated = costs.copy()
    gated[gated > threshold] = threshold + 1e-5
    rows, columns = linear_sum_assignment(gated)
    expected_matches = []
    expected_tracks = [value for index, value in enumerate(track_indices) if index not in rows]
    expected_detections = [value for index, value in enumerate(detection_indices) if index not in columns]
    for row, column in zip(rows, columns, strict=True):
        if gated[row, column] > threshold:
            expected_tracks.append(track_indices[row])
            expected_detections.append(detection_indices[column])
        else:
            expected_matches.append((track_indices[row], detection_indices[column]))
    assert actual == (expected_matches, expected_tracks, expected_detections)


def _reference_cosine(galleries: dict, queries: np.ndarray, targets: list[int]) -> np.ndarray:
    """Reproduce the original raw-gallery cosine calculation per identity."""
    costs = np.zeros((len(targets), len(queries)))
    for row, target in enumerate(targets):
        gallery = np.asarray(galleries[target])
        x = gallery / np.linalg.norm(gallery, axis=1, keepdims=True)
        y = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        costs[row] = (1.0 - np.dot(x, y.T)).min(axis=0)
    return costs


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("budget", [None, 2])
def test_shared_query_normalization_preserves_raw_gallery_updates_and_eviction(dtype, budget) -> None:
    """Distance remains exact through irregular galleries, pruning and edits."""
    rng = np.random.default_rng(45)
    metric = NearestNeighborDistanceMetric("cosine", 0.4, budget=budget)
    samples = rng.normal(size=(9, 17)).astype(dtype)
    samples_before = samples.copy()
    metric.partial_fit(samples, np.array([7, 3, 3, 5, 7, 3, 5, 7, 7]), [7, 3, 5])
    queries = rng.normal(size=(6, 17)).astype(dtype)
    original = queries.copy()
    for targets in ([7, 5, 3], [3, 3, 7], []):
        np.testing.assert_array_equal(
            metric.distance(queries, targets), _reference_cosine(metric.samples, queries, targets)
        )
    np.testing.assert_array_equal(samples, samples_before)
    np.testing.assert_array_equal(queries, original)
    # Public gallery values remain raw and live; normalization must not make
    # subsequent edits invisible or change the number of remembered samples.
    metric.samples[7][-1][0] += 2.345
    metric.partial_fit(rng.normal(size=(2, 17)).astype(dtype), np.array([7, 3]), [3, 7])
    assert set(metric.samples) == {3, 7}
    if budget is not None:
        assert all(len(gallery) <= budget for gallery in metric.samples.values())
    np.testing.assert_array_equal(metric.distance(queries, [3, 7]), _reference_cosine(metric.samples, queries, [3, 7]))


def test_zero_embeddings_and_custom_metric_retain_existing_semantics() -> None:
    """Do not silently redefine zero-vector cosine or normalize custom inputs."""
    metric = NearestNeighborDistanceMetric("cosine", 0.4)
    metric.partial_fit(np.array([[0.0, 0.0], [3.0, 4.0]]), np.array([1, 2]), [1, 2])
    queries = np.array([[0.0, 0.0], [4.0, 3.0]])
    with np.errstate(invalid="ignore", divide="ignore"):
        np.testing.assert_array_equal(
            metric.distance(queries, [2, 1]), _reference_cosine(metric.samples, queries, [2, 1])
        )
    metric._metric = lambda gallery, features: np.asarray(gallery).sum() + np.asarray(features).sum(axis=1)
    np.testing.assert_array_equal(metric.distance(queries, [2, 1]), [[7.0, 14.0], [0.0, 7.0]])


class _CountingDetection(_Detection):
    """Expose costly custom geometry while counting each requested conversion."""

    reads = 0

    @property
    def xyxy(self) -> np.ndarray:
        self.reads += 1
        return super().xyxy

    @property
    def xywha(self) -> np.ndarray:
        self.reads += 1
        return super().xywha


@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("kind", ["xywh", "xyah", "xysr", "xyhr", "xyscr"])
@pytest.mark.parametrize("only_position", [False, True])
def test_motion_preparation_preserves_costs_and_extracts_detections_once(is_obb, kind, only_position) -> None:
    """Hoisting geometry preserves selected-filter gates and OBB alignment."""
    model = create_motion_model(kind, is_obb=is_obb)
    detections = [
        _CountingDetection(
            np.array([40 + offset, 50, 20, 30, 0.23] if is_obb else [30 + offset, 35, 20, 30]),
            0.9, 0, index, np.array([1.0, 0.0]), is_obb=is_obb,
        )
        for index, offset in enumerate([0.0, 2.0, 70.0])
    ]
    tracks = [
        Track(det, index + 1, 1, 30, 10, is_obb=is_obb, motion_model=model) for index, det in enumerate(detections)
    ]
    Track.multi_predict(tracks)
    if is_obb:
        detections[0].tlwh[2:4] = detections[0].tlwh[2:4][::-1]
        detections[0].tlwh[4] += np.pi / 2
    track_indices, detection_indices = [2, 0], [2, 1, 0]
    costs = np.array([[0.25, np.inf, 0.34], [0.45, 0.12, 0.67]])
    expected, blocked = costs.copy(), np.zeros(costs.shape, dtype=bool)
    for row, index in enumerate(track_indices):
        boxes = np.asarray([det.xywha if is_obb else det.xyxy for det in reversed(detections)])
        distances = motion_gating_distance(tracks[index], boxes, only_position=only_position)
        blocked[row] = ~np.isfinite(costs[row]) | ~np.isfinite(distances)
        blocked[row] |= distances > motion_gating_threshold(tracks[index], only_position=only_position)
        keep = ~blocked[row]
        expected[row, keep] = 0.98 * costs[row, keep] + (1 - 0.98) * distances[keep]
        expected[row, ~keep] = INFTY_COST
    for detection in detections:
        detection.reads = 0
    actual_mask = np.zeros_like(blocked)
    actual = gate_cost_matrix(
        costs.copy(), tracks, detections, track_indices, detection_indices, 0.98,
        only_position=only_position, gated_mask=actual_mask,
    )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_mask, blocked)
    assert [detection.reads for detection in detections] == [1, 1, 1]


@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_native_geometry_batch_matches_scalar_rounding_and_declines_custom_tracks(is_obb, dtype) -> None:
    """Default-model geometry keeps ownership, dtype, and extensible accessors."""
    detections = [
        _Detection(
            np.array([42.123456, 53.345678, 17.456789, 23.567891, 0.234567] if is_obb
                     else [33.123456, 41.345678, 17.456789, 23.567891]),
            0.9, 0, index, np.array([1.0, 0.0]), is_obb=is_obb,
        )
        for index in range(3)
    ]
    tracks = [Track(det, index + 1, 1, 30, 10, is_obb=is_obb) for index, det in enumerate(detections)]
    for track in tracks:
        track.mean = track.mean.astype(dtype)
    for cls, items in ((_Detection, detections), (Track, tracks)):
        reference = np.asarray([item.xywha if is_obb else item.xyxy for item in items])
        geometry = cls.batch_geometry(items, is_obb=is_obb)
        np.testing.assert_array_equal(geometry, reference)
        assert geometry.dtype == reference.dtype
        geometry[:] = 0
        np.testing.assert_array_equal(cls.batch_geometry(items, is_obb=is_obb), reference)
    tracks[0].to_tlbr = lambda: np.array([1.0, 2.0, 3.0, 4.0])
    assert Track.batch_geometry(tracks, is_obb=is_obb) is None
    assert _CountingDetection.batch_geometry(detections, is_obb=is_obb) is None
