"""Shared LAP boundary handling preserves hard gates and rejection costs."""

import numpy as np
import pytest

from boxmot.trackers.common.association.matching import linear_assignment, solve_assignment


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("threshold", (-1.0, 0.0, 0.5, 1.0))
@pytest.mark.parametrize("shape", ((1, 1), (2, 4), (4, 2)))
def test_assignment_accepts_exact_cost_limit(dtype, threshold: float, shape: tuple[int, int]) -> None:
    """An eligible boundary edge wins its tie with leaving both ends unmatched."""
    costs = np.full(shape, np.inf, dtype=dtype)
    count = min(shape)
    costs[np.arange(count), np.arange(count)] = threshold

    matches, unmatched_rows, unmatched_columns = linear_assignment(costs, threshold)

    np.testing.assert_array_equal(matches, np.column_stack((np.arange(count), np.arange(count))))
    np.testing.assert_array_equal(unmatched_rows, np.arange(count, shape[0]))
    np.testing.assert_array_equal(unmatched_columns, np.arange(count, shape[1]))


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("threshold", (-1.0, 0.0, 0.5, 1.0))
def test_assignment_rejects_one_step_above_original_limit(dtype, threshold: float) -> None:
    """The internal rejection penalty must never relax the caller's hard gate."""
    above = np.nextafter(dtype(threshold), dtype(np.inf))
    costs = np.array([[threshold, np.inf], [np.inf, above]], dtype=dtype)

    matches, unmatched_rows, unmatched_columns = linear_assignment(costs, threshold)

    np.testing.assert_array_equal(matches, [[0, 0]])
    np.testing.assert_array_equal(unmatched_rows, [1])
    np.testing.assert_array_equal(unmatched_columns, [1])


def test_float32_cost_does_not_round_the_original_limit_up() -> None:
    """A double-precision cutoff remains strict when costs have lower precision."""
    threshold = np.nextafter(np.float64(1.0), -np.inf)
    matches = solve_assignment(np.array([[1.0]], dtype=np.float32), cost_limit=threshold)
    assert matches.shape == (0, 2)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_assignment_preserves_finite_rejection_penalty(dtype) -> None:
    """A cheap single match may beat two eligible but expensive matches."""
    costs = np.array([[0.0, 0.9], [0.9, np.inf]], dtype=dtype)

    matches, unmatched_rows, unmatched_columns = linear_assignment(costs, 1.0)

    np.testing.assert_array_equal(matches, [[0, 0]])
    np.testing.assert_array_equal(unmatched_rows, [1])
    np.testing.assert_array_equal(unmatched_columns, [1])


@pytest.mark.parametrize("threshold", (None, 1.0))
def test_forbidden_edges_cannot_steal_an_eligible_column(threshold: float | None) -> None:
    """Nonfinite costs stay forbidden, with or without an explicit cutoff."""
    costs = np.array([[np.nan, -np.inf, np.inf], [0.1, 0.2, 0.3]])
    original = costs.copy()

    np.testing.assert_array_equal(solve_assignment(costs, cost_limit=threshold), [[1, 0]])
    np.testing.assert_array_equal(costs, original)
