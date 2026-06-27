from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectionRecord:
    """Canonical detection passed from layout parsing into tracker logic."""

    box: np.ndarray
    conf: float
    cls: int
    det_ind: int
    emb: np.ndarray | None = None
    mask: np.ndarray | None = None


@dataclass(frozen=True)
class TrackRecord:
    """Canonical snapshot of a tracker-local track."""

    box: np.ndarray
    track_id: int
    conf: float
    cls: int
    det_ind: int
    state: str
    age: int
    time_since_update: int


@dataclass(frozen=True)
class AssociationResult:
    """Shared association result shape for tracker matching code."""

    matches: np.ndarray
    unmatched_dets: np.ndarray
    unmatched_tracks: np.ndarray
    cost_matrix: np.ndarray | None = None

    @classmethod
    def from_tuple(cls, result: tuple) -> AssociationResult:
        """Adapt association tuples to the shared result record."""
        if len(result) == 3:
            matches, unmatched_dets, unmatched_tracks = result
            cost_matrix = None
        elif len(result) == 4:
            matches, unmatched_dets, unmatched_tracks, cost_matrix = result
        else:
            raise ValueError(f"Unsupported association tuple length: {len(result)}")

        return cls(
            matches=np.asarray(matches, dtype=int).reshape(-1, 2),
            unmatched_dets=np.asarray(unmatched_dets, dtype=int),
            unmatched_tracks=np.asarray(unmatched_tracks, dtype=int),
            cost_matrix=None if cost_matrix is None else np.asarray(cost_matrix),
        )
