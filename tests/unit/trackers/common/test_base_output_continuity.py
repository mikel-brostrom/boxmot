"""The shared tracker boundary preserves scalar OBB continuity and ownership."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from boxmot.structures import Detections, OrientedBoxes, Tracks
from boxmot.trackers.common.box.base import BoxTracker
from boxmot.trackers.common.geometry.obb import align_obb_measurement


class _OutputTracker(BoxTracker):
    """Expose controlled kernel rows while exercising the complete public wrapper."""

    def __init__(self) -> None:
        super().__init__(is_obb=True, min_hits=1, max_age=2, max_obs=5)
        self.kernel_rows = np.empty((0, 9), dtype=np.float32)
        self.mutate_input = False
        self.seen: np.ndarray | None = None

    def _track_detections(
        self,
        dets: np.ndarray,
        img: np.ndarray | None,
        embs: np.ndarray | None = None,
        masks: np.ndarray | None = None,
    ) -> np.ndarray:
        """Optionally modify private inputs and return owned, unformatted rows."""
        self.seen = dets
        if self.mutate_input:
            dets[:, 0] += 1000.0
        return self.kernel_rows


def _input(rows: np.ndarray, *, packed: bool, frame: int) -> Detections | np.ndarray:
    """Use independent valid detections, including explicit empty sensor batches."""
    if packed:
        return rows
    return Detections(
        geometry=OrientedBoxes(torch.from_numpy(np.ascontiguousarray(rows[:, :5]))),
        scores=torch.from_numpy(np.ascontiguousarray(rows[:, 5])),
        class_ids=torch.from_numpy(rows[:, 6].astype(np.int64)),
        sample_id=f"continuity/{frame}",
    )


def _detections(count: int) -> np.ndarray:
    """Make valid geometry that need not equal a kernel's predicted tracks."""
    values = np.tile(np.array([20, 30, 12, 24, 0.1, 0.9, 2], dtype=np.float32), (count, 1))
    values[:, 0] += np.arange(count) * 30
    return values


def _rows(ids: list[int], boxes: list[list[float]], *, dtype=np.float64) -> np.ndarray:
    """Keep emitted row order explicit and preserve unrounded kernel geometry."""
    if not ids:
        return np.empty((0, 9), dtype=dtype)
    count = len(ids)
    return np.column_stack((boxes, ids, np.full(count, 0.9), np.full(count, 2), np.arange(count))).astype(dtype)


def _scalar_geometry(rows: np.ndarray, previous: dict[int, np.ndarray]) -> np.ndarray:
    """Reproduce the former sequential loop, including both float32 roundings."""
    raw = np.asarray(rows, dtype=np.float32)
    geometry = np.ascontiguousarray(raw[:, :5]).copy()
    for index, identity in enumerate(raw[:, 5].astype(np.int64).tolist()):
        reference = previous.get(identity)
        if reference is not None:
            geometry[index] = align_obb_measurement(geometry[index], reference)
        previous[identity] = geometry[index].copy()
    return geometry


def _geometry(output: Tracks | np.ndarray) -> np.ndarray:
    """Expose public geometry storage without making a defensive test copy."""
    return output[:, :5] if isinstance(output, np.ndarray) else output.geometry.values.numpy()


def _ids(output: Tracks | np.ndarray) -> np.ndarray:
    """Read IDs without changing their row order."""
    return output[:, 5].astype(np.int64) if isinstance(output, np.ndarray) else output.track_ids.numpy()


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("kernel_dtype", [np.float32, np.float64])
def test_obb_continuity_matches_scalar_with_reordered_missing_new_and_empty_tracks(packed, kernel_dtype):
    """Equivalent forms and angle wraps align by identity across empty frames."""
    tracker = _OutputTracker()
    history: dict[int, np.ndarray] = {}
    sequence = [
        _rows(
            [41, 7, 99],
            [
                [10.123456789, 20.987654321, 36.0000029, 11.0000011, np.pi - 0.00600000017],
                [40, 50, 14, 43, -np.pi + 0.004],
                [90, 10, 25, 25, np.pi / 4],
            ],
            dtype=kernel_dtype,
        ),
        _rows(
            [99, 41, 123],
            [
                [91, 11, 25, 25, np.pi / 4 + np.pi / 2],
                [10.223456789, 20.887654321, 11.0000011, 36.0000029, -np.pi / 2 + 0.00700000017],
                [150, 60, 20, 8, 12.345678901],
            ],
            dtype=kernel_dtype,
        ),
        _rows([], [], dtype=kernel_dtype),
        _rows(
            [7, 123, 41, 501],
            [
                [41, 49, 43, 14, np.pi / 2 - 0.009],
                [151, 61, 8, 20, 12.345678901 - np.pi / 2],
                [10.323456789, 20.787654321, 36.0000029, 11.0000011, -np.pi + 0.008],
                [200, 70, 40, 9, -15.123456789],
            ],
            dtype=kernel_dtype,
        ),
    ]
    for frame, rows in enumerate(sequence):
        tracker.kernel_rows = rows
        original = rows.copy()
        expected = _scalar_geometry(rows, history)
        output = tracker.update(_input(_detections(len(rows)), packed=packed, frame=frame))
        np.testing.assert_array_equal(_geometry(output), expected)
        np.testing.assert_array_equal(_ids(output), rows[:, 5])
        np.testing.assert_array_equal(rows, original)
        assert tracker._obb_output_by_track_id.keys() == history.keys()
        for identity in history:
            np.testing.assert_array_equal(tracker._obb_output_by_track_id[identity], history[identity])
        if packed:
            assert output.dtype == np.float64 and output.flags.c_contiguous
            assert output.shape == (len(rows), 9)
        else:
            assert output.geometry.values.dtype == torch.float32
            assert output.sample_id == f"continuity/{frame}"


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("kernel_dtype", [np.float32, np.float64])
def test_large_obb_batches_match_scalar_across_equivalent_forms_and_rounding(packed, kernel_dtype):
    """Mixed old/new identities retain exact scalar candidate choice and dtype."""
    rng = np.random.default_rng(8247)
    count = 256
    identities = rng.choice(np.arange(1, 10000), size=count, replace=False)
    centers = rng.uniform(-1e4, 1e4, size=(count, 2))
    sizes = np.exp(rng.uniform(-13, 10, size=(count, 2)))
    sizes[::7, 1] = sizes[::7, 0]  # Square boxes exercise equal-size candidates.
    sizes[1::13] = [1e-8, 7e-7]  # Matched-row clamping must agree after float32 conversion.
    angles = rng.uniform(-40 * np.pi, 40 * np.pi, size=count)
    angles[:8] = [np.pi, -np.pi, np.pi / 2, -np.pi / 2, 0, np.pi / 4, -np.pi / 4, 2 * np.pi]
    offsets = np.array([0, np.pi, np.pi / 2, -np.pi / 2])
    history: dict[int, np.ndarray] = {}
    tracker = _OutputTracker()
    for frame in range(12):
        selected = rng.choice(count, size=128, replace=False)
        forms = rng.integers(0, 4, size=len(selected))
        chosen_sizes = sizes[selected].copy()
        swapped = forms >= 2
        chosen_sizes[swapped] = chosen_sizes[swapped, ::-1]
        geometry = np.column_stack(
            (centers[selected] + frame * 0.123456789, chosen_sizes, angles[selected] + offsets[forms] + frame * 1e-5)
        )
        tracker.kernel_rows = _rows(identities[selected].tolist(), geometry.tolist(), dtype=kernel_dtype)
        original = tracker.kernel_rows.copy()
        expected = _scalar_geometry(original, history)
        output = tracker.update(_input(_detections(len(selected)), packed=packed, frame=frame))
        np.testing.assert_array_equal(_geometry(output), expected)
        np.testing.assert_array_equal(_ids(output), identities[selected])
        np.testing.assert_array_equal(tracker.kernel_rows, original)
        assert tracker._obb_output_by_track_id.keys() == history.keys()
        for identity, reference in history.items():
            np.testing.assert_array_equal(tracker._obb_output_by_track_id[identity], reference)


@pytest.mark.parametrize("packed", [False, True])
def test_obb_continuity_cache_and_public_geometry_have_independent_storage(packed):
    """Caller/kernel/output mutation must not rewrite the next frame's reference."""
    tracker = _OutputTracker()
    original_input = _detections(2)
    input_snapshot = original_input.copy()
    tracker.mutate_input = True
    tracker.kernel_rows = _rows([4, 8], [[20, 30, 12, 24, 3.13], [70, 40, 40, 18, -3.12]], dtype=np.float32)
    raw_snapshot = tracker.kernel_rows.copy()
    expected_cache: dict[int, np.ndarray] = {}
    expected = _scalar_geometry(raw_snapshot, expected_cache)
    input_batch = _input(original_input, packed=packed, frame=0)
    input_geometry = original_input[:, :5] if packed else input_batch.geometry.values.numpy()
    output = tracker.update(input_batch)
    geometry = _geometry(output)
    np.testing.assert_array_equal(original_input, input_snapshot)
    np.testing.assert_array_equal(input_geometry, input_snapshot[:, :5])
    np.testing.assert_array_equal(tracker.kernel_rows, raw_snapshot)
    assert tracker.seen is not None
    np.testing.assert_array_equal(tracker.seen[:, 0], input_snapshot[:, 0] + 1000)
    for cached in tracker._obb_output_by_track_id.values():
        assert not np.shares_memory(cached, original_input)
        assert not np.shares_memory(cached, input_geometry)
        assert not np.shares_memory(cached, tracker.kernel_rows)
        assert not np.shares_memory(cached, geometry)

    input_geometry[:] += 30
    tracker.kernel_rows[:, :5] += 40
    np.testing.assert_array_equal(geometry, expected)
    geometry[0] += 50
    for identity, cached in tracker._obb_output_by_track_id.items():
        np.testing.assert_array_equal(cached, expected_cache[identity])

    tracker.kernel_rows = _rows([8, 4], [[71, 41, 18, 40, -3.12 + np.pi / 2], [21, 31, 24, 12, 3.13 - np.pi / 2]])
    following = tracker.update(_input(_detections(2), packed=packed, frame=1))
    np.testing.assert_array_equal(_geometry(following), _scalar_geometry(tracker.kernel_rows, expected_cache))
    public_snapshot = _geometry(following).copy()
    tracker._obb_output_by_track_id[8][:] += 60
    np.testing.assert_array_equal(_geometry(following), public_snapshot)


@pytest.mark.parametrize("packed", [False, True])
def test_reset_discards_obb_continuity_for_reused_ids(packed):
    """The same numeric identity begins with its raw orientation after reset."""
    tracker = _OutputTracker()
    tracker.kernel_rows = _rows([4], [[20, 30, 12, 24, 0.2]])
    tracker.update(_input(_detections(1), packed=packed, frame=0))
    tracker.reset()
    assert tracker._obb_output_by_track_id == {}
    tracker.kernel_rows = _rows([4], [[20, 30, 24, 12, 0.2 + np.pi / 2]])
    output = tracker.update(_input(_detections(1), packed=packed, frame=1))
    np.testing.assert_array_equal(_geometry(output), tracker.kernel_rows[:, :5].astype(np.float32))


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize(
    "invalid,message",
    [
        ("shape", "shape"),
        ("nonfinite", "non-finite"),
        ("fractional_id", "non-integer"),
        ("negative_id", "negative|non-negative"),
        ("duplicate_id", "duplicate|unique"),
        ("unknown_class", "unknown class"),
        ("detection_index", "detection index"),
        ("score", "range"),
    ],
)
def test_shared_output_continuity_does_not_accept_invalid_kernel_rows(packed, invalid, message):
    """Batch alignment cannot bypass public output shape, identity or value checks."""
    tracker = _OutputTracker()
    rows = _rows([4, 8], [[20, 30, 12, 24, 0.2], [70, 40, 40, 18, -3.12]])
    if invalid == "shape":
        rows = rows[:, :-1]
    elif invalid == "nonfinite":
        rows[0, 4] = np.nan
    elif invalid == "fractional_id":
        rows[0, 5] = 4.5
    elif invalid == "negative_id":
        rows[0, 5] = -1
    elif invalid == "duplicate_id":
        rows[1, 5] = rows[0, 5]
    elif invalid == "unknown_class":
        rows[0, 7] = 3
    elif invalid == "detection_index":
        rows[0, 8] = 2
    else:
        rows[0, 6] = 1.1
    tracker.kernel_rows = rows
    with pytest.raises(ValueError, match=message):
        tracker.update(_input(_detections(2), packed=packed, frame=0))
