"""Mask support preselection preserves the ordered connected-component rule."""

from __future__ import annotations

import numpy as np
import pytest

from boxmot.trackers.maf_hda.association import mask_merge_groups


def _reference_groups(masks: list[np.ndarray], classes: list[int], threshold: float) -> list[list[int]]:
    """Apply the original ordered union rule using direct full-mask intersections."""
    parents = list(range(len(masks)))

    def root(index: int) -> int:
        while parents[index] != index:
            index = parents[index]
        return index

    for i, first in enumerate(masks):
        yi, xi = np.nonzero(first)
        for j in range(i + 1, len(masks)):
            if classes[i] != classes[j]:
                continue
            yj, xj = np.nonzero(masks[j])
            if xi.max() < xj.min() or xj.max() < xi.min() or yi.max() < yj.min() or yj.max() < yi.min():
                continue
            intersection = int((first & masks[j]).sum())
            if intersection / int((first | masks[j]).sum()) >= threshold:
                parents[root(j)] = root(i)
    groups: dict[int, list[int]] = {}
    for index in range(len(masks)):
        groups.setdefault(root(index), []).append(index)
    return list(groups.values())


@pytest.mark.parametrize("threshold", [0.0, 0.2, 0.6, 1.0])
@pytest.mark.parametrize("count", [0, 1, 12])
def test_mask_candidate_preselection_keeps_component_order(threshold: float, count: int) -> None:
    """Classes, disconnected support, holes, and transitive links keep their groups."""
    rng = np.random.default_rng(31)
    masks = []
    for _ in range(count):
        mask = np.zeros((64, 80), dtype=bool)
        x, y = rng.integers(0, 35, size=2)
        mask[y:y + 24, x:x + 30] = rng.random((24, 30)) > 0.2
        masks.append(mask)
    classes = rng.integers(0, 2, size=count).tolist()
    assert mask_merge_groups(masks, classes, threshold) == _reference_groups(masks, classes, threshold)


def test_mask_intersections_are_limited_to_same_class_overlapping_support(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disjoint or differently classified masks never reach pixel intersection."""
    masks = []
    for x in (0, 3, 40, 6):
        mask = np.zeros((20, 60), dtype=bool)
        mask[2:15, x:x + 12] = True
        masks.append(mask)
    original = np.count_nonzero
    calls: list[int] = []

    def record(values: np.ndarray) -> int:
        calls.append(values.size)
        return original(values)

    monkeypatch.setattr(np, "count_nonzero", record)
    assert mask_merge_groups(masks, [0, 0, 0, 1], 0.3) == [[0, 1], [2], [3]]
    assert len(calls) == 1
