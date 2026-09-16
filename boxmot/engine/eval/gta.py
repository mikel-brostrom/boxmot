"""Join cached appearance observations to offline GTA tracklet association."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

import numpy as np

from boxmot.postprocessing.gta import (
    Tracklet,
    get_spatial_constraints,
    merge_tracklets_batched,
    split_tracklets,
)

if TYPE_CHECKING:
    from boxmot.datasets import DatasetSample


# Existing GTA splitter/connector defaults, with a spatial gate spanning 30%
# of the observed sequence extent. Persist these values with evaluation results.
GTA_PARAMETERS = {
    "split_eps": 0.7,
    "split_min_samples": 10,
    "split_max_clusters": 3,
    "split_min_length": 100,
    "merge_distance": 0.4,
    "merge_batch_size": 50,
    "spatial_factor": 0.3,
}


def _validate_rows(rows: np.ndarray) -> None:
    """Require canonical finite AABB MOT9 rows before joining cached detections."""
    if not isinstance(rows, np.ndarray) or rows.ndim != 2 or rows.shape[1] != 9:
        raise ValueError("GTA requires AABB MOT9 rows with detection indices.")
    if rows.dtype.kind not in "if" or not np.isfinite(rows).all():
        raise ValueError("GTA rows must contain finite numeric values.")
    integers = rows[:, (0, 1, 7, 8)]
    if not np.equal(integers, np.floor(integers)).all() or (integers >= np.iinfo(np.int64).max).any():
        raise ValueError("GTA frame, track, class, and detection indices must be exact int64 values.")
    if (rows[:, 0] < 1).any() or (rows[:, 1] < 0).any() or (rows[:, 7] < 0).any() or (rows[:, 8] < -1).any():
        raise ValueError(
            "GTA requires positive frame numbers, nonnegative identities/classes, and detection indices >= -1."
        )
    if len({(int(row[0]), int(row[1])) for row in rows}) != len(rows):
        raise ValueError("GTA input contains duplicate track identities within a frame.")


def _join_features(
    rows: np.ndarray,
    dataset: Iterable[DatasetSample],
    *,
    progress_fn: Callable[[str, int, int | None], None] | None = None,
) -> dict[int, np.ndarray]:
    """Read only referenced embeddings through each frame's aligned detection batch."""
    requested: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row[8] >= 0:
            requested[int(row[0])].append(index)
    total = sum(len(indices) for indices in requested.values())
    if progress_fn is not None:
        progress_fn("Join embeddings", 0, total)
    if not requested:
        return {}

    features: dict[int, np.ndarray] = {}
    sequence: tuple[str, str] | None = None
    seen_frames: set[int] = set()
    dimension: int | None = None
    for sample in dataset:
        identity = (sample.split, sample.sequence_id)
        if sequence is not None and identity != sequence:
            raise ValueError("GTA requires a dataset scoped to exactly one sequence and split.")
        sequence = identity
        frame_number = sample.frame_index + 1
        if frame_number in seen_frames:
            raise ValueError(f"GTA dataset contains duplicate frame index {sample.frame_index}.")
        seen_frames.add(frame_number)
        indices = requested.pop(frame_number, ())
        if not indices:
            continue
        detections = sample.detections
        if detections.sample_id != sample.sample_id:
            raise ValueError("GTA sample and detection identities do not match.")
        if detections.is_obb:
            raise ValueError("GTA supports AABB detections only.")
        embeddings = detections.embeddings
        if embeddings is None:
            raise ValueError(f"GTA requires cached ReID embeddings for sample {sample.sample_id!r}.")
        values = embeddings.detach().cpu().numpy()
        if values.ndim != 2 or values.shape[0] != len(detections) or values.shape[1] < 1:
            raise ValueError(f"GTA embeddings must align with detections for sample {sample.sample_id!r}.")
        if dimension is not None and values.shape[1] != dimension:
            raise ValueError("GTA requires a consistent embedding dimension across the sequence.")
        dimension = values.shape[1]
        for index in indices:
            detection_index = int(rows[index, 8])
            if detection_index >= len(detections):
                raise ValueError(
                    f"GTA detection index {detection_index} is out of range for sample {sample.sample_id!r}."
                )
            feature = np.asarray(values[detection_index], dtype=np.float64)
            norm = np.linalg.norm(feature)
            if not np.isfinite(feature).all() or not np.isfinite(norm) or norm <= 0:
                raise ValueError(f"GTA requires finite, nonzero embeddings for sample {sample.sample_id!r}.")
            features[index] = (feature / norm).astype(np.float32)
        if progress_fn is not None:
            progress_fn("Join embeddings", len(features), total)
    if requested:
        raise ValueError(f"GTA could not find cached samples for frame numbers {sorted(requested)}.")
    return features


def associate_track_rows(
    rows: np.ndarray,
    dataset: Iterable[DatasetSample],
    *,
    progress_fn: Callable[[str, int, int | None], None] | None = None,
) -> np.ndarray:
    """Associate one sequence's AABB MOT9 rows while changing only track IDs.

    Rows contain ``frame, id, x, y, width, height, score, class, detection_index``.
    Frames are one-based; ``DatasetSample.frame_index`` is zero-based. The
    dataset must expose the same aligned detection batches used during replay,
    with cached embeddings loaded. Row order and all other values are retained.

    GTA runs independently per class. Unmatched rows (detection index -1) follow
    the nearest observed row of their original class/track, preferring the
    earlier frame on ties. They participate in temporal exclusion without
    contributing fabricated embeddings. Entirely unmatched tracks remain
    distinct and are not associated. Final IDs are unique across all classes.

    ``progress_fn`` receives phase labels and actual completed/total work.
    Class and batch labels distinguish successive phases. Candidate merging
    has an unknown total (``None``) until the pass completes. No progress is
    written to the console by this function or the underlying algorithms.
    """
    _validate_rows(rows)
    output = rows.copy()
    if not len(rows):
        if progress_fn is not None:
            progress_fn("Join embeddings", 0, 0)
        return output
    features = _join_features(rows, dataset, progress_fn=progress_fn)
    grouped: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        grouped[int(row[7])][int(row[1])].append(index)

    next_id = 1
    for class_id, tracks in sorted(grouped.items()):

        def class_progress(phase: str, completed: int, total: int | None, *, current_class: int = class_id) -> None:
            if progress_fn is not None:
                progress_fn(f"Class {current_class}: {phase}", completed, total)

        observed: dict[int, Tracklet] = {}
        for original_id, indices in sorted(tracks.items()):
            tracklet = Tracklet(original_id)
            for index in sorted(indices, key=lambda index: rows[index, 0]):
                if index in features:
                    row = rows[index]
                    tracklet.append(
                        int(row[0]),
                        float(row[6]),
                        row[2:6].tolist(),
                        class_id,
                        features[index],
                        observation_index=index,
                    )
            if tracklet.times:
                observed[original_id] = tracklet

        split = split_tracklets(
            observed,
            eps=GTA_PARAMETERS["split_eps"],
            min_samples=GTA_PARAMETERS["split_min_samples"],
            max_k=GTA_PARAMETERS["split_max_clusters"],
            len_thres=GTA_PARAMETERS["split_min_length"],
            progress_fn=class_progress if progress_fn is not None else None,
        )
        owners = {index: tracklet for tracklet in split.values() for index in tracklet.observation_indices}
        unmatched_anchors: dict[int, int] = {}
        for indices in tracks.values():
            matched = [index for index in indices if index in features]
            if not matched:
                continue
            for index in indices:
                if index not in features:
                    nearest = min(
                        matched, key=lambda candidate: (abs(rows[candidate, 0] - rows[index, 0]), rows[candidate, 0])
                    )
                    unmatched_anchors[index] = nearest
                    owners[nearest].unmatched_times.add(int(rows[index, 0]))

        if split:
            max_x_range, max_y_range = get_spatial_constraints(split, factor=GTA_PARAMETERS["spatial_factor"])
            merged = merge_tracklets_batched(
                split,
                batch_size=GTA_PARAMETERS["merge_batch_size"],
                max_x_range=max_x_range,
                max_y_range=max_y_range,
                merge_dist_thres=GTA_PARAMETERS["merge_distance"],
                progress_fn=class_progress if progress_fn is not None else None,
            )
            for _, tracklet in sorted(merged.items()):
                output[tracklet.observation_indices, 1] = next_id
                next_id += 1
        for index, anchor in unmatched_anchors.items():
            output[index, 1] = output[anchor, 1]
        for _, indices in sorted(tracks.items()):
            if not any(index in features for index in indices):
                output[indices, 1] = next_id
                next_id += 1
    return output


__all__ = ("GTA_PARAMETERS", "associate_track_rows")
