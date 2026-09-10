"""Regression coverage for observation history shared by SORT trackers."""

from __future__ import annotations

import numpy as np
import pytest

from boxmot.trackers.common.tracking.observations import k_previous_obs, speed_direction


@pytest.mark.parametrize(("is_obb", "width"), [(False, 5), (True, 6)])
def test_empty_history_returns_the_geometry_specific_missing_observation(is_obb: bool, width: int) -> None:
    np.testing.assert_array_equal(k_previous_obs({}, cur_age=4, k=3, is_obb=is_obb), [-1] * width)


@pytest.mark.parametrize(("ages", "expected_age"), [((9, 7, 8), 7), ((9, 8), 8)])
def test_history_selects_the_oldest_available_observation_in_the_lookback_window(
    ages: tuple[int, ...], expected_age: int
) -> None:
    observations = {age: np.array([age, 0, age + 10, 20, 0.9]) for age in ages}

    selected = k_previous_obs(observations, cur_age=10, k=3)

    np.testing.assert_array_equal(selected, observations[expected_age])


def test_history_falls_back_to_latest_observation_after_a_long_gap() -> None:
    observations = {5: np.array([5, 0, 15, 20, 0.9]), 2: np.array([2, 0, 12, 20, 0.9])}

    selected = k_previous_obs(observations, cur_age=10, k=3)

    np.testing.assert_array_equal(selected, observations[5])


def test_aabb_direction_uses_center_motion_and_returns_dy_before_dx() -> None:
    first = np.array([0, 0, 10, 20, 0.9])
    second = np.array([1, 0, 15, 28, 0.6])

    direction = speed_direction(first, second)

    np.testing.assert_allclose(direction, [0.8, 0.6], atol=1e-6)


def test_aabb_direction_stays_finite_when_size_changes_without_center_motion() -> None:
    first = np.array([0, 0, 10, 20, 0.9])
    second = np.array([-2, -4, 12, 24, 0.6])

    direction = speed_direction(first, second)

    np.testing.assert_array_equal(direction, [0.0, 0.0])
