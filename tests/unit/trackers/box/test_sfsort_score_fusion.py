"""SFSORT confidence fusion affects only high-confidence association."""

from __future__ import annotations

import numpy as np
import pytest

from boxmot import SFSORT, SFSORTConfig


def _rows(is_obb: bool, confidence: float, *, offset: float = 0.0) -> np.ndarray:
    """Provide one equivalent square in either geometry layout."""
    box = [20 + offset, 20, 20, 20, 0] if is_obb else [10 + offset, 10, 30 + offset, 30]
    return np.array([[*box, confidence, 0]], dtype=np.float64)


@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("fuse", [False, True])
@pytest.mark.parametrize("confidence", [0.3, 0.65])
def test_score_fusion_changes_high_pass_but_preserves_low_pass(is_obb: bool, fuse: bool, confidence: float) -> None:
    """Confidence can reject a first-pass match without altering recovery costs."""
    tracker = SFSORT(
        SFSORTConfig(
            high_th=0.5, new_track_th=0.5, match_th_first=0.2, fuse_detection_score=fuse,
            frame_width=100, frame_height=100,
        ),
        is_obb=is_obb,
    )
    first = tracker.update(_rows(is_obb, 0.99))
    second = tracker.update(_rows(is_obb, confidence))
    assert len(first) == len(second) == 1
    id_column = 5 if is_obb else 4
    assert (first[0, id_column] == second[0, id_column]) == (not fuse or confidence < 0.5)


@pytest.mark.parametrize("is_obb", [False, True])
def test_score_fusion_preserves_centroid_radius(is_obb: bool) -> None:
    """Low confidence cannot bring a geometrically forbidden match into range."""
    tracker = SFSORT(
        SFSORTConfig(
            high_th=0.5, new_track_th=0.5, fuse_detection_score=True,
            asso_func="centroid", centroid_max_distance=0.01, frame_width=100, frame_height=100,
        ),
        is_obb=is_obb,
    )
    first = tracker.update(_rows(is_obb, 0.99))
    second = tracker.update(_rows(is_obb, 0.65, offset=5.0))
    id_column = 5 if is_obb else 4
    assert first[0, id_column] != second[0, id_column]
