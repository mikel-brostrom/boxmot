"""Validation of independent packed 3D observations at the tracker boundary."""

from __future__ import annotations

import numpy as np
import pytest

from boxmot.trackers.common.input import parse_numpy_detection_rows_3d


@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64])
@pytest.mark.parametrize("count", [0, 1, 3])
def test_spatial_rows_preserve_readonly_strided_inputs(dtype, count: int) -> None:
    """Geometry, confidence, and exact classes survive normalization independently."""
    storage = np.tile([1, 2, -10, 0, 4, 2, 2, 1, 2**40], (count * 2, 1)).astype(dtype)
    detections = storage[::-2]
    original = detections.copy()
    detections.flags.writeable = False

    parsed = parse_numpy_detection_rows_3d(detections)

    assert parsed.geometry.shape == (count, 7)
    assert parsed.geometry.dtype == np.float32
    assert parsed.scores.shape == (count,)
    assert parsed.scores.dtype == np.float32
    assert parsed.class_ids.shape == (count,)
    assert parsed.class_ids.dtype == np.int64
    assert all(values.flags.c_contiguous for values in (parsed.geometry, parsed.scores, parsed.class_ids))
    np.testing.assert_array_equal(parsed.geometry, original[:, :7])
    np.testing.assert_array_equal(parsed.scores, original[:, 7])
    np.testing.assert_array_equal(parsed.class_ids, original[:, 8])
    np.testing.assert_array_equal(detections, original)
    assert not np.shares_memory(parsed.class_ids, detections)


@pytest.mark.parametrize("depth", [0, -10, 10])
def test_spatial_depth_is_not_a_visibility_requirement(depth: float) -> None:
    """Off-camera observations remain valid input for the fusion algorithm."""
    parsed = parse_numpy_detection_rows_3d(np.array([[1, 2, depth, 0, 4, 2, 2, 0.9, 0]]))
    assert parsed.geometry[0, 2] == depth


@pytest.mark.parametrize("shape", [(9,), (1, 8), (1, 10), (0, 8), (1, 1, 9)])
def test_spatial_rows_require_exact_nine_column_shape(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match=r"3D detection rows must have shape \[N, 9\]"):
        parse_numpy_detection_rows_3d(np.zeros(shape))


@pytest.mark.parametrize("dtype", [object, np.complex64, np.bool_, np.str_])
def test_spatial_rows_reject_nonreal_dtypes(dtype) -> None:
    with pytest.raises(TypeError, match="real numeric dtype"):
        parse_numpy_detection_rows_3d(np.zeros((1, 9), dtype=dtype))


def test_spatial_rows_reject_lists_and_numpy_subclasses() -> None:
    class _Array(np.ndarray):
        """A subclass that must not change public boundary behavior."""

    for detections in ([[1, 2, 3, 0, 4, 2, 2, 0.9, 0]], np.zeros((1, 9)).view(_Array)):
        with pytest.raises(TypeError, match="detections_3d must be Detections3D or a plain numpy.ndarray"):
            parse_numpy_detection_rows_3d(detections)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        (0, np.nan, "finite values"),
        (2, np.inf, "finite values"),
        (7, np.nan, "finite values"),
        (8, np.inf, "finite values"),
        (0, 1e100, "representable as finite float32"),
        (4, 0, "positive length, width, and height"),
        (5, -1, "positive length, width, and height"),
        (6, 0, "positive length, width, and height"),
        (4, 1e-100, "positive length, width, and height"),
        (7, -0.1, r"inclusive range \[0, 1\]"),
        (7, 1.1, r"inclusive range \[0, 1\]"),
        (8, -1, "non-negative integers"),
        (8, 0.5, "non-negative integers"),
        (8, 2**63, "non-negative integers"),
        (8, 2**53 + 2, "exact float64 integer range"),
    ],
)
@pytest.mark.parametrize("count", [1, 32])
def test_invalid_spatial_values_fail_without_mutating_input(
    column: int, value: float, message: str, count: int
) -> None:
    detections = np.tile([1, 2, 10, 0, 4, 2, 2, 0.9, 0], (count, 1))
    detections[-1, column] = value
    original = detections.copy()
    with pytest.raises(ValueError, match=message):
        parse_numpy_detection_rows_3d(detections)
    np.testing.assert_array_equal(detections, original)


@pytest.mark.parametrize("count", [1, 32])
@pytest.mark.parametrize(
    ("dtype", "class_id", "message"),
    [
        (np.float64, 2**40 + 1, None),
        (np.float64, 2**53, None),
        (np.int64, 2**53, None),
        (np.uint64, 2**53, None),
        (np.int64, 2**53 + 1, "exact float64 integer range"),
        (np.uint64, 2**53 + 1, "exact float64 integer range"),
        (np.uint64, 2**63, "non-negative integers"),
    ],
)
def test_spatial_class_ids_are_validated_before_float32_conversion(
    dtype, class_id: int, message: str | None, count: int
) -> None:
    detections = np.tile(np.array([[1, 2, 10, 0, 4, 2, 2, 1, class_id]], dtype=dtype), (count, 1))
    if message is not None:
        with pytest.raises(ValueError, match=message):
            parse_numpy_detection_rows_3d(detections)
        return

    parsed = parse_numpy_detection_rows_3d(detections)
    np.testing.assert_array_equal(parsed.class_ids, np.full(count, class_id, dtype=np.int64))
