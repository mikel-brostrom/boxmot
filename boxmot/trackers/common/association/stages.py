from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

from boxmot.trackers.common.association.matching import linear_assignment as matching_linear_assignment
from boxmot.trackers.common.tracking.records import AssociationResult

TrackT = TypeVar("TrackT")
DetectionT = TypeVar("DetectionT")

IndexSelector = Callable[[Sequence], np.ndarray | Sequence[int]]
CostBuilder = Callable[[Sequence[TrackT], Sequence[DetectionT]], np.ndarray]
AssignmentSolver = Callable[[np.ndarray, float], tuple[np.ndarray, Sequence[int], Sequence[int]]]
StageMatcher = Callable[
    [Sequence[TrackT], Sequence[DetectionT]],
    AssociationResult | tuple[np.ndarray, Sequence[int], Sequence[int]],
]
DetectionTrackAssignmentSolver = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class AssociationStage:
    """Declarative description of one tracker association pass."""

    name: str
    threshold: float
    cost: CostBuilder | None = None
    matcher: StageMatcher | None = None
    track_selector: IndexSelector | None = None
    detection_selector: IndexSelector | None = None
    update_on_match: bool = True
    create_unmatched_detections: bool = False


@dataclass(frozen=True)
class AssociationStageResult:
    """Association result plus the selected source indices for a stage."""

    stage: AssociationStage
    association: AssociationResult
    track_indices: np.ndarray
    detection_indices: np.ndarray

    @property
    def matches(self) -> np.ndarray:
        return self.association.matches

    @property
    def unmatched_tracks(self) -> np.ndarray:
        return self.association.unmatched_tracks

    @property
    def unmatched_dets(self) -> np.ndarray:
        return self.association.unmatched_dets

    @property
    def cost_matrix(self) -> np.ndarray | None:
        return self.association.cost_matrix

    def absolute_matches(self) -> np.ndarray:
        """Return matches mapped back to the original track/detection indices."""
        if self.matches.size == 0:
            return np.empty((0, 2), dtype=int)
        return np.column_stack(
            (
                self.track_indices[self.matches[:, 0]],
                self.detection_indices[self.matches[:, 1]],
            )
        ).astype(int, copy=False)

    def absolute_unmatched_tracks(self) -> np.ndarray:
        return self.track_indices[self.unmatched_tracks].astype(int, copy=False)

    def absolute_unmatched_dets(self) -> np.ndarray:
        return self.detection_indices[self.unmatched_dets].astype(int, copy=False)


def all_indices(items: Sequence) -> np.ndarray:
    """Select every item in a track or detection collection."""
    return np.arange(len(items), dtype=int)


def run_association_stage(
    stage: AssociationStage,
    tracks: Sequence[TrackT],
    detections: Sequence[DetectionT],
    *,
    assignment_solver: AssignmentSolver = matching_linear_assignment,
) -> AssociationStageResult:
    """Build costs and solve one association stage with canonical bookkeeping."""
    track_indices = _resolve_indices(stage.track_selector, tracks)
    detection_indices = _resolve_indices(stage.detection_selector, detections)
    selected_tracks = _take_items(tracks, track_indices)
    selected_detections = _take_items(detections, detection_indices)

    if stage.matcher is not None:
        matched = stage.matcher(selected_tracks, selected_detections)
        association = (
            matched
            if isinstance(matched, AssociationResult)
            else AssociationResult(
                matches=np.asarray(matched[0], dtype=int).reshape(-1, 2),
                unmatched_tracks=np.asarray(matched[1], dtype=int),
                unmatched_dets=np.asarray(matched[2], dtype=int),
            )
        )
    else:
        if stage.cost is None:
            raise ValueError(f"Association stage {stage.name!r} requires cost or matcher")
        cost_matrix = np.asarray(stage.cost(selected_tracks, selected_detections))
        matches, unmatched_tracks, unmatched_dets = assignment_solver(
            cost_matrix,
            stage.threshold,
        )
        association = AssociationResult(
            matches=np.asarray(matches, dtype=int).reshape(-1, 2),
            unmatched_dets=np.asarray(unmatched_dets, dtype=int),
            unmatched_tracks=np.asarray(unmatched_tracks, dtype=int),
            cost_matrix=cost_matrix,
        )
    return AssociationStageResult(
        stage=stage,
        association=association,
        track_indices=track_indices,
        detection_indices=detection_indices,
    )


def _resolve_indices(selector: IndexSelector | None, items: Sequence) -> np.ndarray:
    selected = all_indices(items) if selector is None else selector(items)
    return np.asarray(selected, dtype=int).reshape(-1)


def _take_items(items: Sequence[TrackT], indices: np.ndarray) -> Sequence[TrackT]:
    if isinstance(items, np.ndarray):
        return items[indices]
    return [items[int(i)] for i in indices]


def detection_track_iou_assignment(
    iou_matrix: np.ndarray,
    threshold: float,
    assignment_solver: DetectionTrackAssignmentSolver,
) -> AssociationResult:
    """Adapt detector-row/track-column IoU matching to canonical stage output."""
    iou_matrix = np.asarray(iou_matrix, dtype=float)
    n_dets, n_tracks = iou_matrix.shape
    if iou_matrix.size == 0 or iou_matrix.max(initial=0.0) <= threshold:
        return AssociationResult(
            matches=np.empty((0, 2), dtype=int),
            unmatched_tracks=np.arange(n_tracks, dtype=int),
            unmatched_dets=np.arange(n_dets, dtype=int),
            cost_matrix=1.0 - iou_matrix.T,
        )

    matched_indices = np.asarray(assignment_solver(-iou_matrix), dtype=int).reshape(-1, 2)
    matches = []
    matched_dets = set()
    matched_tracks = set()
    for det_idx, track_idx in matched_indices:
        if iou_matrix[det_idx, track_idx] < threshold:
            continue
        matches.append([track_idx, det_idx])
        matched_dets.add(int(det_idx))
        matched_tracks.add(int(track_idx))

    unmatched_dets = [idx for idx in range(n_dets) if idx not in matched_dets]
    unmatched_tracks = [idx for idx in range(n_tracks) if idx not in matched_tracks]
    return AssociationResult(
        matches=np.asarray(matches, dtype=int).reshape(-1, 2),
        unmatched_tracks=np.asarray(unmatched_tracks, dtype=int),
        unmatched_dets=np.asarray(unmatched_dets, dtype=int),
        cost_matrix=1.0 - iou_matrix.T,
    )


def detection_track_tuple_to_association_result(
    result: tuple[np.ndarray, Sequence[int], Sequence[int]],
    cost_matrix: np.ndarray | None = None,
) -> AssociationResult:
    """Convert detector-row/track-column tuples to canonical track-row orientation."""
    matches, unmatched_dets, unmatched_tracks = result
    matches = np.asarray(matches, dtype=int).reshape(-1, 2)
    if matches.size > 0:
        matches = matches[:, [1, 0]]
    return AssociationResult(
        matches=matches,
        unmatched_tracks=np.asarray(unmatched_tracks, dtype=int),
        unmatched_dets=np.asarray(unmatched_dets, dtype=int),
        cost_matrix=cost_matrix,
    )
