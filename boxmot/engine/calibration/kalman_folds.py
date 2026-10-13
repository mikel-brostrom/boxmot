"""Validate and partition complete trajectories for held-out Kalman selection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from boxmot.engine.calibration.kalman_data import CalibrationData, CalibrationTrack


def track_key(track: CalibrationTrack) -> tuple[str, int, int]:
    """Identify complete trajectories without relying on input ordering."""
    return track.sequence_id, int(track.track_id), int(track.class_id)


def track_reference(track: CalibrationTrack) -> dict[str, Any]:
    """Serialize the identity partition so experiments can be audited."""
    return {"sequence_id": track.sequence_id, "track_id": int(track.track_id), "class_id": int(track.class_id)}


def matched_detections(track: CalibrationTrack) -> np.ndarray:
    """Identify observed boxes without consulting their GT geometry."""
    return np.all(np.isfinite(track.detection_boxes), axis=1)


def usable_tracks(data: CalibrationData, geometry: str) -> tuple[CalibrationTrack, ...]:
    """Validate inputs and retain trajectories with an actual prediction target."""
    width = 5 if geometry == "obb" else 4
    usable = []
    seen = set()
    for track in sorted(data.tracks, key=track_key):
        if track_key(track) in seen:
            raise ValueError("KF selection requires unique sequence/identity/class trajectories.")
        seen.add(track_key(track))
        frames = np.asarray(track.frame_indices)
        if (
            frames.ndim != 1
            or not np.isfinite(frames).all()
            or np.any(frames != np.floor(frames))
            or np.any(np.diff(frames) <= 0)
            or track.gt_boxes.shape != (len(frames), width)
            or track.detection_boxes.shape != (len(frames), width)
            or track.scores.shape != (len(frames),)
        ):
            raise ValueError("KF selection requires aligned boxes and strictly increasing delivered-frame indices.")
        truth = np.asarray(track.gt_boxes)
        sizes = truth[:, 2:4] if geometry == "obb" else truth[:, 2:4] - truth[:, :2]
        if not np.isfinite(truth).all() or np.any(sizes <= 0):
            raise ValueError("KF selection requires finite ground-truth boxes with positive dimensions.")
        observed = matched_detections(track)
        boxes = track.detection_boxes[observed]
        sizes = boxes[:, 2:4] if geometry == "obb" else boxes[:, 2:4] - boxes[:, :2]
        scores = track.scores[observed]
        if np.any(sizes <= 0) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError("KF selection requires positive detection boxes and confidence in [0, 1].")
        if np.any(observed[:-1] & (np.diff(frames) == 1)):
            usable.append(track)
    if len(usable) < 2:
        raise ValueError(
            "KF selection requires at least two usable trajectories, each with a matched detection "
            "followed by an adjacent annotated frame. Select more annotated data."
        )
    return tuple(usable)


def grouped_folds(
    tracks: Sequence[CalibrationTrack],
) -> tuple[str, list[tuple[tuple[CalibrationTrack, ...], tuple[CalibrationTrack, ...]]]]:
    """Hold out entire sequences, falling back to identities in one sequence."""
    sequences = sorted({track.sequence_id for track in tracks})
    grouping = "sequence" if len(sequences) > 1 else "trajectory"
    groups = sequences if grouping == "sequence" else sorted({track_key(track)[:2] for track in tracks})
    if len(groups) < 2:
        raise ValueError("KF selection requires at least two independent sequences or trajectory identities.")
    assignments = {key: index % min(3, len(groups)) for index, key in enumerate(groups)}
    folds = []
    for index in range(min(3, len(groups))):
        validation, training = [], []
        for track in tracks:
            key = track.sequence_id if grouping == "sequence" else track_key(track)[:2]
            (validation if assignments[key] == index else training).append(track)
        if sum(int(matched_detections(track).sum()) for track in training) < 2:
            raise ValueError("KF selection requires at least two matched detections in every training fold.")
        folds.append((tuple(training), tuple(validation)))
    return grouping, folds


def validate_camera_candidates(
    candidates: Mapping[str, Mapping[str, np.ndarray] | None], tracks: Sequence[CalibrationTrack]
) -> dict[str, dict[str, np.ndarray] | None]:
    """Require an explicit finite previous-to-current affine at every used frame."""
    if not candidates:
        raise ValueError("Joint CMC/KF selection requires at least one camera policy candidate.")
    validated = {}
    last_frames: dict[str, int] = {}
    for track in tracks:
        if np.any(track.frame_indices < 0):
            raise ValueError("Camera transforms require nonnegative delivered-frame indices.")
        last_frames[track.sequence_id] = max(last_frames.get(track.sequence_id, 0), int(track.frame_indices[-1]))
    for name, sequences in candidates.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Camera policy candidate names must be nonempty strings.")
        if sequences is None:
            validated[name] = None
            continue
        if not isinstance(sequences, Mapping):
            raise ValueError(f"Camera policy {name!r} requires sequence-indexed transforms or None.")
        warps = {}
        for sequence_id, last_frame in last_frames.items():
            if sequence_id not in sequences:
                raise ValueError(f"Camera policy {name!r} has no transforms for sequence {sequence_id!r}.")
            matrices = np.asarray(sequences[sequence_id], dtype=np.float64)
            if matrices.ndim != 3 or matrices.shape[1:] != (2, 3) or len(matrices) <= last_frame:
                raise ValueError(
                    f"Camera policy {name!r}, sequence {sequence_id!r}: transforms must have shape "
                    f"(N, 2, 3) and cover delivered frame {last_frame}."
                )
            if not np.isfinite(matrices).all():
                raise ValueError(f"Camera policy {name!r}, sequence {sequence_id!r}: transforms must be finite.")
            if np.any(np.abs(np.linalg.det(matrices[:, :, :2])) < 1e-10):
                raise ValueError(f"Camera policy {name!r}, sequence {sequence_id!r}: transforms must be nonsingular.")
            warps[sequence_id] = matrices
        validated[name] = warps
    return validated
