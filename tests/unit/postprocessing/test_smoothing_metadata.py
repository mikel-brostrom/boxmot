"""Interpolation and smoothing preserve canonical MOT row metadata."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
import pytest

from boxmot.postprocessing.gbrc import gradient_boosting_smooth
from boxmot.postprocessing.gbrc import linear_interpolation as gbrc_interpolation
from boxmot.postprocessing.gbrc import process_file as process_gbrc_file
from boxmot.postprocessing.gsi import gaussian_smooth
from boxmot.postprocessing.gsi import linear_interpolation as gsi_interpolation
from boxmot.postprocessing.gsi import process_file as process_gsi_file

INTERPOLATORS = (gsi_interpolation, gbrc_interpolation)
SMOOTHERS = (
    partial(gaussian_smooth, tau=3.0),
    partial(gradient_boosting_smooth, n_estimators=5, min_samples_split=2),
)


@pytest.mark.parametrize("interpolate", INTERPOLATORS, ids=("gsi", "gbrc"))
def test_interpolation_marks_synthetic_detections_and_preserves_classes(interpolate: Callable) -> None:
    """Only continuous box/score values may be interpolated across a gap."""
    rows = np.array(
        [[3, 8, 2.5, 4.5, 6.5, 8.5, 0.9, 4, 12], [1, 8, 0.5, 2.5, 4.5, 6.5, 0.5, 4, 2]],
        dtype=np.float64,
    )
    original = rows.copy()
    result = interpolate(rows, interval=4)

    np.testing.assert_array_equal(result[[0, 2]], original[[1, 0]])
    np.testing.assert_allclose(result[1], [2, 8, 1.5, 3.5, 5.5, 7.5, 0.7, 4, -1])
    np.testing.assert_array_equal(rows, original)


@pytest.mark.parametrize("interpolate", INTERPOLATORS, ids=("gsi", "gbrc"))
def test_interpolation_never_bridges_different_classes_with_the_same_id(interpolate: Callable) -> None:
    rows = np.array([[1, 5, 0, 0, 4, 4, 1, 2, 8], [3, 5, 8, 8, 4, 4, 1, 6, 13]], dtype=np.float64)
    np.testing.assert_array_equal(interpolate(rows, interval=20), rows)


@pytest.mark.parametrize("smooth", SMOOTHERS, ids=("gsi", "gbrc"))
def test_smoothing_keeps_observed_metadata_and_class_trajectories_independent(smooth: Callable) -> None:
    """Reused IDs in different classes must not pull each other's boxes together."""
    first = np.array(
        [[1, 9, 0.25, 0.5, 4.25, 5.5, 0.7, 2, 7], [3, 9, 1.25, 1.5, 4.75, 6.5, 0.9, 2, 11]],
        dtype=np.float64,
    )
    second = first.copy()
    second[:, 2:4] += 100.0
    second[:, 7] = 6
    second[:, 8] = [21, 25]
    rows = np.concatenate((first, second))
    original = rows.copy()
    progress = []

    result = smooth(rows, progress_fn=lambda current, total: progress.append((current, total)))

    for group in (first, second):
        selected = result[result[:, 7] == group[0, 7]]
        np.testing.assert_allclose(selected[:, 2:6], smooth(group)[:, 2:6])
        np.testing.assert_array_equal(selected[:, [0, 1, 6, 7, 8]], group[:, [0, 1, 6, 7, 8]])
    np.testing.assert_array_equal(rows, original)
    assert progress == [(1, 2), (2, 2)]


@pytest.mark.parametrize(
    "operation",
    (partial(gsi_interpolation, interval=20), partial(gbrc_interpolation, interval=20), *SMOOTHERS),
    ids=("gsi-interpolate", "gbrc-interpolate", "gsi-smooth", "gbrc-smooth"),
)
@pytest.mark.parametrize("empty", (False, True))
def test_empty_and_singleton_rows_remain_exact(operation: Callable, empty: bool) -> None:
    rows = np.array([[1, 9, 0.125, 0.375, 4.625, 5.875, 0.73456789, 4, 17]], dtype=np.float64)
    if empty:
        rows = rows[:0]
    original = rows.copy()
    result = operation(rows)
    np.testing.assert_array_equal(result, original)
    np.testing.assert_array_equal(rows, original)
    assert result.shape == rows.shape


@pytest.mark.parametrize(
    "process",
    (
        partial(process_gsi_file, interval=20, tau=3.0),
        partial(process_gbrc_file, interval=20, n_estimators=5, learning_rate=0.065, min_samples_split=2),
    ),
    ids=("gsi", "gbrc"),
)
def test_file_postprocessing_retains_box_precision_and_detection_index(tmp_path: Path, process: Callable) -> None:
    path = tmp_path / "sequence.txt"
    original = np.array([[1, 9, 0.125, 0.375, 4.625, 5.875, 0.73456789, 4, 17]], dtype=np.float64)
    np.savetxt(path, original, delimiter=",", fmt="%.17g")

    process(path)

    np.testing.assert_array_equal(np.loadtxt(path, delimiter=",", ndmin=2), original)
