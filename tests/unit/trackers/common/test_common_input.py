"""Tests for the packed NumPy tracker boundary."""

from __future__ import annotations

import numpy as np
import pytest

from boxmot.trackers.common.input import pack_numpy_track_rows


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
