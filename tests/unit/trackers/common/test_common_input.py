"""Tests for the packed NumPy tracker boundary."""

from __future__ import annotations

import numpy as np
import pytest

from boxmot.trackers.common.input import pack_numpy_track_rows, parse_numpy_detection_rows


def _pack(**overrides) -> np.ndarray:
    values = {
        "geometry": np.array([[1, 2, 11, 22]], dtype=np.float32),
        "track_ids": np.array([7], dtype=np.int64),
        "scores": np.array([0.9], dtype=np.float32),
        "class_ids": np.array([2], dtype=np.int64),
        "detection_indices": np.array([0], dtype=np.int64),
        "is_obb": False,
        "detection_count": 1,
        "owner": "Test tracker",
    }
    values.update(overrides)
    return pack_numpy_track_rows(**values)


def test_pack_numpy_track_rows_returns_contiguous_float64_aabb8() -> None:
    rows = _pack()

    assert rows.dtype == np.float64
    assert rows.shape == (1, 8)
    assert rows.flags.c_contiguous
    np.testing.assert_allclose(rows, [[1, 2, 11, 22, 7, np.float32(0.9), 2, 0]])


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"track_ids": np.array([-1], dtype=np.int64)}, "negative track IDs"),
        (
            {
                "geometry": np.array([[1, 2, 11, 22], [2, 3, 12, 23]], dtype=np.float32),
                "track_ids": np.array([7, 7], dtype=np.int64),
                "scores": np.array([0.9, 0.8], dtype=np.float32),
                "class_ids": np.array([2, 2], dtype=np.int64),
                "detection_indices": np.array([0, 0], dtype=np.int64),
            },
            "duplicate track IDs",
        ),
        ({"scores": np.array([1.1], dtype=np.float32)}, r"range \[0, 1\]"),
        ({"detection_indices": np.array([-2], dtype=np.int64)}, "below -1"),
        ({"detection_indices": np.array([1], dtype=np.int64)}, "outside the current batch"),
        ({"class_ids": np.array([2**53 + 2], dtype=np.int64)}, "exact float64 integer range"),
    ),
)
def test_pack_numpy_track_rows_rejects_invalid_metadata(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _pack(**overrides)


@pytest.mark.parametrize("field", ["track_ids", "class_ids", "detection_indices"])
@pytest.mark.parametrize(
    ("values", "message"),
    [
        (np.array([np.nan]), "non-finite"),
        (np.array([0.5]), "non-integer"),
        (np.array([np.iinfo(np.uint64).max], dtype=np.uint64), "int64 range"),
    ],
)
def test_output_metadata_other_dtypes_are_still_validated(field: str, values: np.ndarray, message: str) -> None:
    """Skipping impossible int64 checks must retain checks for floating and unsigned values."""
    with pytest.raises(ValueError, match=message):
        _pack(**{field: values})


def test_int64_metadata_keeps_packed_precision_limit() -> None:
    """Native integer arrays still require exact float64 representation."""
    with pytest.raises(ValueError, match="exact float64 integer range"):
        _pack(class_ids=np.array([np.iinfo(np.int64).max], dtype=np.int64))


@pytest.mark.parametrize("class_id", [-1, 0.5, float(2**63)])
def test_detection_class_ids_keep_integer_range_checks(class_id: float) -> None:
    """A cached integer limit must preserve validation before class conversion."""
    detections = np.array([[1, 2, 11, 22, 0.9, class_id]])
    with pytest.raises(ValueError, match="non-negative integers"):
        parse_numpy_detection_rows(detections, is_obb=False)
