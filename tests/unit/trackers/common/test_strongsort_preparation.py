"""StrongSORT preparation keeps assignment order and appearance/motion costs."""

import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment

from boxmot import StrongSortConfig
from boxmot.structures import Boxes, Detections
from boxmot.trackers.common.association.matching import motion_gating_distance, motion_gating_threshold
from boxmot.trackers.common.association.strongsort import (
    INFTY_COST,
    NearestNeighborDistanceMetric,
    gate_cost_matrix,
    min_cost_matching,
)
from boxmot.trackers.common.motion.models import create_motion_model
from boxmot.trackers.strongsort.track import Track, _Detection
from boxmot.trackers.strongsort.tracker import StrongSort


@pytest.mark.parametrize("shape", [(2, 5), (5, 2), (4, 4), (1, 1), (0, 3), (3, 0)])
@pytest.mark.parametrize("tied", [False, True])
def test_unmatched_masks_preserve_original_partition_order(shape, tied: bool) -> None:
    """Rectangular, tied and rejected assignments keep caller index ordering."""
    costs = np.random.default_rng(734).uniform(size=shape)
    if tied:
        costs[:] = 0.5
    if costs.size:
        costs.flat[0] = 0.5  # An edge exactly at the threshold remains admissible.
    threshold = 0.5
    actual = min_cost_matching(costs.copy(), threshold)
    gated = costs.copy()
    gated[gated > threshold] = threshold + 1e-5
    rows, columns = linear_sum_assignment(gated)
    expected_matches = []
    expected_tracks = [index for index in range(shape[0]) if index not in rows]
    expected_detections = [index for index in range(shape[1]) if index not in columns]
    for row, column in zip(rows, columns, strict=True):
        if gated[row, column] > threshold:
            expected_tracks.append(row)
            expected_detections.append(column)
        else:
            expected_matches.append((row, column))
    assert actual == (expected_matches, expected_tracks, expected_detections)


@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (0, 0)])
def test_empty_matching_returns_unmatched_local_indices_as_lists(shape: tuple[int, int]) -> None:
    """Empty assignments retain every row or column in a consistent list contract."""
    actual = min_cost_matching(np.empty(shape), 0.5)
    assert all(isinstance(partition, list) for partition in actual)
    assert actual == ([], list(range(shape[0])), list(range(shape[1])))


def test_geometric_recovery_preserves_candidate_order_after_rejected_appearance_match(monkeypatch) -> None:
    """An unassigned appearance row must not displace the earlier rejected track."""

    def observations(count: int) -> Detections:
        """Give coincident boxes distinct, valid appearance embeddings."""
        return Detections(
            geometry=Boxes(torch.tensor([[10.0, 10.0, 30.0, 30.0]] * count)),
            scores=torch.full((count,), 0.9),
            class_ids=torch.zeros(count, dtype=torch.int64),
            embeddings=torch.eye(3)[:count],
        )

    tracker = StrongSort(StrongSortConfig(use_cmc=False, n_init=1, max_cos_dist=0.5, max_iou_dist=0.5))
    initial = tracker.update(observations(3))
    original_tracks = list(tracker.active_tracks)
    monkeypatch.setattr(
        tracker, "_appearance_cost", lambda tracks, detections: np.array([[0.8, 0.9], [0.9, 0.8], [0.1, 0.2]])
    )
    monkeypatch.setattr(
        tracker, "_association_cost", lambda tracks, detections: np.full((len(tracks), len(detections)), 0.1)
    )

    # Appearance matches track 2, leaves track 1 unassigned, and rejects track 0.
    # Geometric matching must still consider track 0 before track 1 for the tie.
    actual = tracker.update(observations(2))

    assert actual.track_ids.tolist() == [initial.track_ids[0].item(), initial.track_ids[2].item()]
    assert actual.detection_indices.tolist() == [1, 0]
    assert tracker.active_tracks == original_tracks
    assert original_tracks[1].time_since_update == 1


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
def test_euclidean_gallery_distance_returns_closest_squared_distance(dtype) -> None:
    """Each query selects the nearest NumPy sample within the requested gallery."""
    metric = NearestNeighborDistanceMetric("euclidean", 20.0)
    metric.partial_fit(
        np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]], dtype=dtype),
        np.array([7, 7, 3]),
        [7, 3],
    )
    queries = np.array([[1.0, 0.0], [4.0, 4.0], [0.0, 3.0]], dtype=dtype)

    np.testing.assert_array_equal(metric.distance(queries, [3, 7]), [[10.0, 17.0, 0.0], [1.0, 16.0, 9.0]])


def test_euclidean_gallery_distance_accepts_empty_queries() -> None:
    """No queries produces an empty cost column for every requested target."""
    metric = NearestNeighborDistanceMetric("euclidean", 20.0)
    metric.partial_fit(np.array([[0.0, 0.0], [4.0, 0.0]]), np.array([7, 3]), [7, 3])

    costs = metric.distance(np.empty((0, 2)), [3, 7])

    assert isinstance(costs, np.ndarray)
    assert costs.shape == (2, 0)


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
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_motion_preparation_preserves_costs_and_extracts_detections_once(is_obb, kind, only_position, dtype) -> None:
    """Hoisting geometry preserves selected-filter gates and OBB alignment."""
    model = create_motion_model(kind, is_obb=is_obb)
    detections = [
        _CountingDetection(
            np.array([40 + offset, 50, 20, 30, 0.23] if is_obb else [30 + offset, 35, 20, 30]),
            0.9,
            0,
            index,
            np.array([1.0, 0.0]),
            is_obb=is_obb,
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
    selected_tracks = [tracks[2], tracks[0]]
    selected_detections = list(reversed(detections))
    costs = np.array([[0.25, np.inf, 0.34], [0.45, 0.12, 0.67]], dtype=dtype)
    expected, blocked = costs.copy(), np.zeros(costs.shape, dtype=bool)
    for row, track in enumerate(selected_tracks):
        boxes = np.asarray([det.xywha if is_obb else det.xyxy for det in selected_detections])
        distances = motion_gating_distance(track, boxes, only_position=only_position)
        blocked[row] = ~np.isfinite(costs[row]) | ~np.isfinite(distances)
        blocked[row] |= distances > motion_gating_threshold(track, only_position=only_position)
        keep = ~blocked[row]
        expected[row, keep] = 0.98 * costs[row, keep] + (1 - 0.98) * distances[keep]
        expected[row, ~keep] = INFTY_COST
    for detection in detections:
        detection.reads = 0
    actual_mask = np.zeros_like(blocked)
    actual = gate_cost_matrix(
        costs.copy(),
        selected_tracks,
        selected_detections,
        0.98,
        only_position=only_position,
        gated_mask=actual_mask,
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
            np.array(
                [42.123456, 53.345678, 17.456789, 23.567891, 0.234567]
                if is_obb
                else [33.123456, 41.345678, 17.456789, 23.567891]
            ),
            0.9,
            0,
            index,
            np.array([1.0, 0.0]),
            is_obb=is_obb,
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
