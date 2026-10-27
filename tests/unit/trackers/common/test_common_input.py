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


@pytest.mark.parametrize(
    ("dtype", "class_id", "error"),
    [
        (np.int8, 127, None),
        (np.uint8, 255, None),
        (np.int16, 32767, None),
        (np.uint16, 65535, None),
        (np.int32, 2**31 - 1, None),
        (np.uint32, 2**32 - 1, None),
        (np.int64, 2**53, None),
        (np.uint64, 2**53, None),
        (np.int64, 2**53 + 1, "exact float64 integer range"),
        (np.uint64, 2**53 + 1, "exact float64 integer range"),
        (np.int64, 2**63 - 1, "exact float64 integer range"),
        (np.uint64, 2**63 - 1, "exact float64 integer range"),
        (np.uint64, 2**63, "non-negative integers"),
        (np.uint64, 2**64 - 1, "non-negative integers"),
        (np.int64, -(2**63), "non-negative integers"),
        (np.float16, np.finfo(np.float16).max, None),
        (np.float16, 0.5, "non-negative integers"),
        (np.float32, 2**53, None),
        (np.float32, np.nextafter(np.float32(2**53), np.float32(np.inf)), "exact float64 integer range"),
        (np.float32, 2**63, "non-negative integers"),
        (np.float64, 2**53, None),
        # This integer has already rounded to 2**53 when stored as float64.
        (np.float64, 2**53 + 1, None),
        (np.float64, 2**53 + 2, "exact float64 integer range"),
        (np.float64, np.nextafter(float(2**63), 0), "exact float64 integer range"),
        (np.float64, 2**63 - 1, "non-negative integers"),
        (np.float64, 2**63, "non-negative integers"),
        (np.float64, -0.0, None),
    ],
)
@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("count", [1, 32])
def test_detection_class_limits_preserve_source_dtype(
    dtype, class_id, error: str | None, is_obb: bool, count: int
) -> None:
    """Integer comparisons must not round large uint64 classes through float64."""
    geometry = [1, 2, 11, 22, 0] if is_obb else [1, 2, 11, 22]
    detections = np.tile(np.array([[*geometry, 1, class_id]], dtype=dtype), (count, 1))
    if error is not None:
        with pytest.raises(ValueError, match=error):
            parse_numpy_detection_rows(detections, is_obb=is_obb)
        return

    with np.errstate(over="raise", invalid="raise"):
        parsed = parse_numpy_detection_rows(detections, is_obb=is_obb)
    np.testing.assert_array_equal(parsed.class_ids, np.full(count, int(detections[0, -1]), dtype=np.int64))
    assert parsed.class_ids.dtype == np.int64


@pytest.mark.parametrize("count", [1, 32])
def test_detection_extended_float_class_precision(count: int) -> None:
    """Extended precision must retain the same integer thresholds as Python ints."""
    if np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant:
        pytest.skip("longdouble has no additional precision on this platform")
    for class_id in (2**53 + 1, 2**63 - 1):
        detections = np.tile(np.array([[1, 2, 11, 22, 1, class_id]], dtype=np.longdouble), (count, 1))
        with pytest.raises(ValueError, match="exact float64 integer range"):
            parse_numpy_detection_rows(detections, is_obb=False)


@pytest.mark.parametrize(
    ("class_ids", "error"),
    [
        ([2**53 + 2, -1], "exact float64 integer range"),
        ([-1, 2**53 + 2], "non-negative integers"),
        ([2**53 + 2, 0.5], "exact float64 integer range"),
        ([0.5, 2**53 + 2], "non-negative integers"),
        ([2**53 + 2, 2**63], "exact float64 integer range"),
        ([2**63, 2**53 + 2], "non-negative integers"),
    ],
)
@pytest.mark.parametrize("count", [3, 35])
def test_detection_class_validation_reports_first_rejected_row(class_ids: list[float], error: str, count: int) -> None:
    detections = np.tile([1, 2, 11, 22, 1, 0], (count, 1)).astype(np.float64)
    detections[-2:, -1] = class_ids
    with pytest.raises(ValueError, match=error):
        parse_numpy_detection_rows(detections, is_obb=False)


@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("count", [0, 1, 3])
def test_detection_parser_keeps_contiguous_arrays_and_owned_classes(is_obb: bool, count: int) -> None:
    geometry = [1, 2, 11, 22, 0] if is_obb else [1, 2, 11, 22]
    storage = np.tile([*geometry, 1, 7], (count * 2, 1)).astype(np.int64)
    detections = storage[::-2]
    parsed = parse_numpy_detection_rows(detections, is_obb=is_obb)

    assert parsed.geometry.shape == (count, len(geometry))
    assert parsed.geometry.dtype == np.float32
    assert parsed.scores.shape == (count,)
    assert parsed.scores.dtype == np.float32
    assert parsed.class_ids.shape == (count,)
    assert parsed.class_ids.dtype == np.int64
    assert all(values.flags.c_contiguous for values in (parsed.geometry, parsed.scores, parsed.class_ids))
    assert not np.shares_memory(parsed.class_ids, detections)
    detections[:, -1] = 9
    np.testing.assert_array_equal(parsed.class_ids, np.full(count, 7, dtype=np.int64))


@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_output_packing_preserves_strided_readonly_values_and_float32_rounding(is_obb: bool, dtype) -> None:
    """Packing strided columns must keep values, ownership, and the public layout."""
    geometry_columns = 5 if is_obb else 4
    geometry = np.tile([1.123456789, 2.345678912, 11.456789123, 22.567891234, 0.123456789], (6, 2))
    geometry = geometry.astype(dtype)[::-2, :geometry_columns]
    scores = np.array([0, 0.123456789, 0, 0.234567891, 0, 0.345678912], dtype=dtype)[::-2]
    track_ids = np.arange(6, dtype=np.int64)[::-2]
    class_ids = np.array([0, 2**53, 0, 2**53 - 1, 0, 2**53 - 2], dtype=np.int64)[::-2]
    detection_indices = np.arange(6, dtype=np.int64)[::-2]
    columns = (geometry, scores, track_ids, class_ids, detection_indices)
    original_values = [values.copy() for values in columns]
    for values in columns:
        values.flags.writeable = False

    rows = _pack(
        geometry=geometry,
        scores=scores,
        track_ids=track_ids,
        class_ids=class_ids,
        detection_indices=detection_indices,
        is_obb=is_obb,
        detection_count=6,
    )

    expected = np.column_stack(
        (geometry.astype(np.float32), track_ids, scores.astype(np.float32), class_ids, detection_indices)
    )
    np.testing.assert_array_equal(rows, expected)
    assert rows.dtype == np.float64
    assert rows.shape == (3, geometry_columns + 4)
    assert rows.flags.c_contiguous
    for values, original in zip(columns, original_values):
        np.testing.assert_array_equal(values, original)
        assert not np.shares_memory(rows, values)


@pytest.mark.parametrize("score", [0.123456789, np.float32(0.123456789), np.array(0.123456789)])
def test_output_packing_retains_scalar_score_coercion(score) -> None:
    rows = _pack(scores=score)
    assert rows[0, 5] == float(np.float32(score))
