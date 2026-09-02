from __future__ import annotations

from collections.abc import Callable

import numpy as np

from boxmot.trackers.common.association.matching import solve_assignment

SimilarityFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]
CornerVelocities = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def confidence_difference(
    detections: np.ndarray,
    tracks: np.ndarray,
    *,
    track_confidence_column: int = -1,
) -> np.ndarray:
    """Return absolute detector/track confidence differences."""
    if len(detections) == 0 or len(tracks) == 0:
        return np.zeros((len(detections), len(tracks)), dtype=float)
    return np.abs(detections[:, 4, np.newaxis] - tracks[np.newaxis, :, track_confidence_column])


def _direction_matrix(
    detections: np.ndarray,
    previous_observations: np.ndarray,
    *,
    x_column: int,
    y_column: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized track-to-detection directions as track-by-detection matrices."""
    dx = detections[:, x_column] - previous_observations[:, x_column, np.newaxis]
    dy = detections[:, y_column] - previous_observations[:, y_column, np.newaxis]
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    return dy / norm, dx / norm


def _velocity_consistency(
    detections: np.ndarray,
    previous_observations: np.ndarray,
    velocities: np.ndarray,
    velocity_weight: float,
    *,
    x_column: int,
    y_column: int,
) -> np.ndarray:
    direction_y, direction_x = _direction_matrix(
        detections,
        previous_observations,
        x_column=x_column,
        y_column=y_column,
    )
    velocity_y = velocities[:, 0, np.newaxis]
    velocity_x = velocities[:, 1, np.newaxis]
    cosine = np.clip(velocity_x * direction_x + velocity_y * direction_y, -1.0, 1.0)
    angle = (np.pi / 2.0 - np.abs(np.arccos(cosine))) / np.pi
    valid = (previous_observations[:, 4] >= 0).astype(float)[:, np.newaxis]
    detection_confidence = detections[:, -1, np.newaxis]
    return (valid * angle * velocity_weight).T * detection_confidence


def _four_corner_motion_cost(
    detections: np.ndarray,
    previous_observations: np.ndarray,
    corner_velocities: CornerVelocities,
    velocity_weight: float,
) -> np.ndarray:
    corners = ((0, 1), (0, 3), (2, 1), (2, 3))
    return sum(
        (
            _velocity_consistency(
                detections,
                previous_observations,
                velocities,
                velocity_weight,
                x_column=x_column,
                y_column=y_column,
            )
            for velocities, (x_column, y_column) in zip(corner_velocities, corners)
        ),
        start=np.zeros((len(detections), len(previous_observations)), dtype=float),
    )


def _geometry_candidates(
    similarity: np.ndarray,
    ranking_similarity: np.ndarray,
    threshold: float,
) -> np.ndarray:
    if min(similarity.shape, default=0) == 0:
        return np.empty((0, 2), dtype=int)

    admissible = similarity > threshold
    if admissible.sum(axis=1).max() == 1 and admissible.sum(axis=0).max() == 1:
        return np.argwhere(admissible)
    return solve_assignment(-ranking_similarity)


def _partition_matches(
    candidates: np.ndarray,
    accepted: Callable[[int, int], bool],
    *,
    detection_count: int,
    track_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matches: list[list[int]] = []
    matched_detections: set[int] = set()
    matched_tracks: set[int] = set()
    for detection_index, track_index in np.asarray(candidates, dtype=int).reshape(-1, 2):
        if not accepted(int(detection_index), int(track_index)):
            continue
        matches.append([int(detection_index), int(track_index)])
        matched_detections.add(int(detection_index))
        matched_tracks.add(int(track_index))

    unmatched_detections = np.asarray(
        [index for index in range(detection_count) if index not in matched_detections],
        dtype=int,
    )
    unmatched_tracks = np.asarray(
        [index for index in range(track_count) if index not in matched_tracks],
        dtype=int,
    )
    return (
        np.asarray(matches, dtype=int).reshape(-1, 2),
        unmatched_detections,
        unmatched_tracks,
    )


def associate_hybrid(
    detections: np.ndarray,
    tracks: np.ndarray,
    similarity_threshold: float,
    corner_velocities: CornerVelocities,
    previous_observations: np.ndarray,
    velocity_weight: float,
    association_function: SimilarityFunction,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Associate HybridSORT detections using geometry, motion, and confidence consistency."""
    if len(tracks) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections), dtype=int),
            np.empty((0,), dtype=int),
        )

    similarity = np.asarray(association_function(detections, tracks), dtype=float)
    ranking_similarity = (
        similarity
        + _four_corner_motion_cost(
            detections,
            previous_observations,
            corner_velocities,
            velocity_weight,
        )
        - confidence_difference(detections, tracks, track_confidence_column=4)
    )
    candidates = _geometry_candidates(similarity, ranking_similarity, similarity_threshold)
    return _partition_matches(
        candidates,
        lambda detection_index, track_index: (similarity[detection_index, track_index] >= similarity_threshold),
        detection_count=len(detections),
        track_count=len(tracks),
    )


def associate_hybrid_with_reid(
    detections: np.ndarray,
    tracks: np.ndarray,
    similarity_threshold: float,
    corner_velocities: CornerVelocities,
    previous_observations: np.ndarray,
    velocity_weight: float,
    association_function: SimilarityFunction,
    *,
    embedding_cost: np.ndarray,
    geometry_weight: float = 1.0,
    embedding_weight: float = 0.0,
    longterm_embedding_cost: np.ndarray | None = None,
    longterm_embedding_weight: float = 0.0,
    correct_with_appearance: bool = False,
    appearance_threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Associate HybridSORT detections with geometry, motion, and appearance."""
    if len(tracks) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections), dtype=int),
            np.empty((0,), dtype=int),
        )

    similarity = np.asarray(association_function(detections, tracks), dtype=float)
    motion_and_confidence = _four_corner_motion_cost(
        detections,
        previous_observations,
        corner_velocities,
        velocity_weight,
    ) - confidence_difference(detections, tracks, track_confidence_column=4)
    assignment_cost = geometry_weight * -(similarity + motion_and_confidence)
    assignment_cost += embedding_weight * embedding_cost
    if longterm_embedding_cost is not None:
        assignment_cost += longterm_embedding_weight * longterm_embedding_cost
    candidates = solve_assignment(assignment_cost)

    threshold_similarity = similarity - confidence_difference(
        detections,
        tracks,
        track_confidence_column=4,
    )

    def accepted(detection_index: int, track_index: int) -> bool:
        geometry_is_weak = threshold_similarity[detection_index, track_index] < similarity_threshold
        if correct_with_appearance:
            appearance_is_weak = embedding_cost[detection_index, track_index] > appearance_threshold
            return not (geometry_is_weak and appearance_is_weak)
        return not geometry_is_weak

    return _partition_matches(
        candidates,
        accepted,
        detection_count=len(detections),
        track_count=len(tracks),
    )
