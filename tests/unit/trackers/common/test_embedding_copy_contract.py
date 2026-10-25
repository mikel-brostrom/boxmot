"""Appearance arithmetic preserves precision and caller-owned feature arrays."""

import numpy as np
import pytest

from boxmot.trackers.common.appearance import blend_embeddings, normalize_embedding


def _embedding(dtype: type, layout: str, *, readonly: bool) -> np.ndarray:
    """Build ordinary, strided and read-only views over the same feature values."""
    values = (np.arange(48).reshape(6, 8) - 17.125).astype(dtype)
    if layout == "contiguous":
        result = values.ravel()
    elif layout == "strided":
        result = values.ravel()[::3]
    elif layout == "reversed":
        result = values.ravel()[::-1]
    elif layout == "fortran":
        result = values.T
    else:
        result = values[::2, ::2].T
    if readonly:
        result.flags.writeable = False
    return result


def _copied_float_array(values: np.ndarray) -> np.ndarray:
    """Reference the previous owned-input arithmetic and dtype conversion."""
    dtype = values.dtype if np.issubdtype(values.dtype, np.floating) else np.float32
    return values.astype(dtype, copy=True)


def _copied_normalization(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Keep the reference reduction on an independently copied input."""
    values = _copied_float_array(values)
    norm = float(np.linalg.norm(values))
    return np.zeros_like(values) if norm <= eps else values / norm


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.int16, np.int64, np.bool_])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "reversed", "fortran", "transposed_slice"])
@pytest.mark.parametrize("readonly", [False, True])
def test_normalization_preserves_exact_values_dtype_and_input_storage(
    dtype: type, layout: str, readonly: bool
) -> None:
    """Removing intermediate copies must retain rounded values and independent output."""
    source = _embedding(dtype, layout, readonly=readonly)
    snapshot = source.copy()
    expected = _copied_normalization(source)

    actual = normalize_embedding(source)

    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected, strict=True)
    np.testing.assert_array_equal(source, snapshot, strict=True)
    assert actual.flags.writeable
    assert not np.shares_memory(actual, source)


@pytest.mark.parametrize(
    "previous_dtype,current_dtype",
    [
        (np.float16, np.float32),
        (np.float32, np.float64),
        (np.int64, np.float64),
        (np.float32, np.int16),
        (np.int16, np.int64),
        (np.float32, np.float32),
    ],
)
@pytest.mark.parametrize("layout", ["contiguous", "strided", "reversed", "fortran", "transposed_slice"])
def test_blend_preserves_mixed_dtype_rounding_and_readonly_inputs(
    previous_dtype: type, current_dtype: type, layout: str
) -> None:
    """Each multiplication retains its operand dtype before the common sum dtype."""
    previous = _embedding(previous_dtype, layout, readonly=True)
    current = _embedding(current_dtype, layout, readonly=True)[::-1]
    previous_snapshot, current_snapshot = previous.copy(), current.copy()
    previous_weight, current_weight = 0.73123456789, 0.26876543211
    expected = _copied_normalization(
        previous_weight * _copied_float_array(previous) + current_weight * _copied_float_array(current)
    )

    actual = blend_embeddings(previous, current, previous_weight, current_weight)

    np.testing.assert_array_equal(actual, expected, strict=True)
    np.testing.assert_array_equal(previous, previous_snapshot, strict=True)
    np.testing.assert_array_equal(current, current_snapshot, strict=True)
    assert actual.flags.writeable
    assert not np.shares_memory(actual, previous)
    assert not np.shares_memory(actual, current)


@pytest.mark.parametrize("values", [[0.0, -0.0], [1e-14, -1e-14], []])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
def test_degenerate_normalization_returns_independent_zeros(values: list[float], dtype: type) -> None:
    """Zero, near-zero and empty inputs retain their existing epsilon behavior."""
    source = np.asarray(values, dtype=dtype)
    source.flags.writeable = False

    actual = normalize_embedding(source)

    np.testing.assert_array_equal(actual, _copied_normalization(source), strict=True)
    assert actual.flags.writeable
    assert not np.shares_memory(actual, source)
