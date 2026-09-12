"""Observation history and motion-direction helpers shared by SORT trackers."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def k_previous_obs(
    observations: Mapping[int, np.ndarray],
    cur_age: int,
    k: int,
    is_obb: bool = False,
) -> np.ndarray | list[int]:
    """Return the oldest available observation in the previous ``k`` updates.

    If the window is empty, return the latest stored observation. An empty
    history returns a missing-observation row containing five values for
    AABB geometry or six for OBB geometry, including confidence.
    """
    if len(observations) == 0:
        return [-1] * (6 if is_obb else 5)
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age - dt]
    max_age = max(observations.keys())
    return observations[max_age]


def speed_direction(bbox1: np.ndarray, bbox2: np.ndarray) -> np.ndarray:
    """Return the normalized AABB center displacement in ``(dy, dx)`` order."""
    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm
