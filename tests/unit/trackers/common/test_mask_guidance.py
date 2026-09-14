from __future__ import annotations

import numpy as np
import pytest

from boxmot.trackers.common.association.masks import apply_mask_guidance
from boxmot.trackers.common.association.matching import linear_assignment


def test_mask_guidance_resolves_crossing_association_without_changing_solver() -> None:
    costs = np.array([[0.35, 0.25], [0.25, 0.35]])
    boxes = np.array([[0, 0, 4, 4], [2, 0, 6, 4]])
    masks = np.zeros((2, 4, 6), dtype=bool)
    masks[0, :2, :2] = True
    masks[1, :2, 4:6] = True

    guided = apply_mask_guidance(costs, boxes, masks, threshold=0.5)
    baseline_matches, _, _ = linear_assignment(costs, 0.5)
    guided_matches, _, _ = linear_assignment(guided, 0.5)

    np.testing.assert_array_equal(baseline_matches, [[0, 1], [1, 0]])
    np.testing.assert_array_equal(guided_matches, [[0, 0], [1, 1]])
    np.testing.assert_allclose(guided, [[0.10, 0.25], [0.25, 0.10]])
    np.testing.assert_array_equal(costs, [[0.35, 0.25], [0.25, 0.35]])


@pytest.mark.parametrize("shape", [(1, 2), (2, 1)])
def test_mask_guidance_handles_row_and_column_ambiguity(shape: tuple[int, int]) -> None:
    costs = np.full(shape, 0.4)
    boxes = np.tile([0, 0, 4, 4], (shape[1], 1))
    masks = np.ones((shape[0], 4, 4), dtype=bool)

    guided = apply_mask_guidance(costs, boxes, masks, threshold=0.4)

    np.testing.assert_allclose(guided, np.full(shape, -0.6))


def test_mask_guidance_accepts_exact_paper_fill_and_coverage_thresholds() -> None:
    # Nine of ten mask pixels lie in a 180-pixel box: mc=0.90, mf=0.05.
    mask = np.zeros((11, 20), dtype=bool)
    mask[0, :9] = True
    mask[10, 0] = True

    guided = apply_mask_guidance(
        np.array([[0.4, 0.4]]),
        np.array([[0, 0, 20, 9], [0, 0, 20, 9]]),
        [mask],
        threshold=0.5,
    )

    np.testing.assert_allclose(guided, [[0.35, 0.35]])


@pytest.mark.parametrize("failed_gate", ["coverage", "fill", "invisible", "missing"])
@pytest.mark.parametrize("costs", [np.array([[0.3, 0.4]]), np.array([[0.95, 0.95]])], ids=["ambiguous", "isolated"])
def test_mask_guidance_preserves_costs_when_mask_gate_fails(failed_gate: str, costs: np.ndarray) -> None:
    mask = np.zeros((10, 10), dtype=bool)
    if failed_gate == "coverage":
        mask[:2, :5] = True
        boxes = np.tile([0, 0, 4, 4], (2, 1))  # mc=0.8, mf=0.5.
    else:
        boxes = np.tile([0, 0, 10, 10], (2, 1))
        if failed_gate == "fill":
            mask[:2, :2] = True  # mc=1.0, mf=0.04.
    guided = apply_mask_guidance(
        costs, boxes, [None if failed_gate == "missing" else mask], threshold=0.5
    )

    np.testing.assert_array_equal(guided, costs)


def test_mask_guidance_uses_positive_logits_for_visibility() -> None:
    logits = np.full((4, 4), -1.0)
    logits[:2, :2] = 0.0
    costs = np.array([[0.3, 0.4]])
    boxes = np.tile([0, 0, 4, 4], (2, 1))

    invisible = apply_mask_guidance(costs, boxes, [logits], threshold=0.5)
    logits[:2, :2] = 1.0
    visible = apply_mask_guidance(costs, boxes, [logits], threshold=0.5)

    np.testing.assert_array_equal(invisible, costs)
    np.testing.assert_allclose(visible, [[0.05, 0.15]])


def test_mask_guidance_preserves_clear_matches_without_cost_penalties() -> None:
    costs = np.array([[0.2, 0.95], [0.95, 0.4]])
    boxes = np.tile([0, 0, 4, 4], (2, 1))
    masks = np.ones((2, 4, 4), dtype=bool)

    guided = apply_mask_guidance(costs, boxes, masks, threshold=0.5)

    np.testing.assert_array_equal(guided, costs)


def test_mask_guidance_recovers_isolation_above_original_assignment_threshold() -> None:
    costs = np.array([[0.95]])

    guided = apply_mask_guidance(
        costs,
        np.array([[0, 0, 4, 4]]),
        [np.ones((4, 4), dtype=bool)],
        threshold=0.9,
    )

    np.testing.assert_allclose(guided, [[-0.05]])
    matches, _, _ = linear_assignment(guided, 0.9)
    np.testing.assert_array_equal(matches, [[0, 0]])


def test_mask_guidance_uses_original_costs_and_current_stage_threshold() -> None:
    costs = np.array([[0.4, 0.8]])
    boxes = np.tile([0, 0, 4, 4], (2, 1))
    masks = [np.ones((4, 4), dtype=bool)]

    low_threshold = apply_mask_guidance(costs, boxes, masks, threshold=0.5)
    high_threshold = apply_mask_guidance(costs, boxes, masks, threshold=0.9)

    np.testing.assert_array_equal(low_threshold, costs)
    np.testing.assert_allclose(high_threshold, [[-0.6, -0.2]])


def test_mask_guidance_clips_actual_xyxy_extent_at_image_boundary() -> None:
    boxes = np.tile([-2.9, -1.9, 2.9, 2.9], (2, 1))
    costs = np.array([[0.4, 0.4]])
    mask = np.ones((4, 5), dtype=bool)

    insufficient_coverage = apply_mask_guidance(costs, boxes, [mask], threshold=0.5)
    np.testing.assert_array_equal(insufficient_coverage, costs)
    mask[3:, :] = False
    mask[:, 3:] = False

    guided = apply_mask_guidance(costs, boxes, [mask], threshold=0.5)

    np.testing.assert_allclose(guided, [[-0.6, -0.6]])


def test_mask_guidance_rasterizes_fractional_boxes_with_floor_and_ceil() -> None:
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True

    guided = apply_mask_guidance(
        np.array([[0.4, 0.4]]), np.tile([1.2, 1.2, 2.2, 2.2], (2, 1)), [mask], threshold=0.5
    )

    np.testing.assert_allclose(guided, [[-0.6, -0.6]])


@pytest.mark.parametrize("box", [[0, 0, 0, 4], [0, 0, 4, 0], [3, 3, 1, 1], [5, 5, 9, 9], [-5, -5, -1, -1]])
def test_mask_guidance_skips_boxes_without_frame_pixels(box: list[int]) -> None:
    costs = np.array([[0.3, 0.4]])

    guided = apply_mask_guidance(
        costs, np.tile(box, (2, 1)), [np.ones((4, 4), dtype=bool)], threshold=0.5
    )

    np.testing.assert_array_equal(guided, costs)


@pytest.mark.parametrize("shape", [(0, 0), (0, 2), (2, 0)])
def test_mask_guidance_preserves_empty_matrix_shape(shape: tuple[int, int]) -> None:
    costs = np.empty(shape)

    guided = apply_mask_guidance(costs, np.empty((shape[1], 4)), [None] * shape[0], threshold=0.5)

    assert guided.shape == shape
    assert guided is not costs


@pytest.mark.parametrize("transpose", [False, True], ids=["isolated_track", "isolated_detection"])
def test_mask_guidance_requires_both_endpoints_to_be_isolated(transpose: bool) -> None:
    costs = np.array([[0.95, 0.95], [0.2, 0.2]])
    if transpose:
        costs = costs.T

    guided = apply_mask_guidance(
        costs, np.tile([0, 0, 4, 4], (2, 1)), np.ones((2, 4, 4), dtype=bool), threshold=0.5
    )

    np.testing.assert_array_equal(guided[costs == 0.95], costs[costs == 0.95])


def test_mask_guidance_protects_clear_pairs_beside_ambiguity_and_isolation() -> None:
    costs = np.array(
        [[0.2, 0.95, 0.95, 0.95], [0.95, 0.3, 0.4, 0.95], [0.95, 0.4, 0.3, 0.95], [0.95, 0.95, 0.95, 0.95]]
    )

    guided = apply_mask_guidance(
        costs, np.tile([0, 0, 4, 4], (4, 1)), np.ones((4, 4, 4), dtype=bool), threshold=0.5
    )

    np.testing.assert_allclose(
        guided,
        [[0.2, 0.95, 0.95, 0.95], [0.95, -0.7, -0.6, 0.95], [0.95, -0.6, -0.7, 0.95], [0.95, 0.95, 0.95, -0.05]],
    )
    matches, _, _ = linear_assignment(guided, 0.5)
    np.testing.assert_array_equal(matches, [[0, 0], [1, 1], [2, 2], [3, 3]])
