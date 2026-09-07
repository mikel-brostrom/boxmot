# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

"""Reusable Global Tracklet Association (GTA) algorithms.

Implements the postprocessing pipeline from:
    Sun et al., "GTA: Global Tracklet Association for Multi-Object Tracking
    in Sports", ACCV 2024 Workshop.

The algorithms operate on complete, in-memory tracklets after tracking and
consist of:

1. **Tracklet Splitter**: Detects identity switches within a single tracklet
   using DBSCAN clustering on ReID embeddings and splits mixed-identity
   tracklets into separate pure-identity tracklets.

2. **Tracklet Connector**: Merges tracklets belonging to the same identity
   using hierarchical agglomerative clustering with average pairwise cosine
   distance (Eq. 1 from the paper), subject to spatial constraints.

BoxMOT v24 deliberately leaves model inference and persisted-artifact joins to
pipelines and the engine.  This module therefore contains no command entry
point, model construction, or positional-cache reader.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm

from boxmot.utils import logger as LOGGER

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Tracklet:
    """Represents a single tracklet with detections, scores, bboxes, and features.

    Attributes:
        track_id: Unique identifier for the track.
        parent_id: Original track ID before any splitting.
        times: Frame numbers where the track is present.
        scores: Detection confidence scores per frame.
        bboxes: Bounding boxes per frame, each [x, y, w, h].
        classes: Class IDs per frame.
        features: L2-normalised ReID embedding vectors per frame.

    Invariant: len(times) == len(scores) == len(bboxes) == len(classes) == len(features).
    """

    track_id: Optional[int] = None
    parent_id: Optional[int] = None
    times: list[int] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    bboxes: list[list[float]] = field(default_factory=list)
    classes: list[int] = field(default_factory=list)
    features: list[np.ndarray] = field(default_factory=list)

    def __init__(
        self,
        track_id: Optional[int] = None,
        frames=None,
        scores=None,
        bboxes=None,
        feats=None,
        classes=None,
    ):
        self.track_id = track_id
        self.parent_id = track_id
        self.scores = (
            scores
            if isinstance(scores, list)
            else [scores]
            if scores is not None
            else []
        )
        self.times = (
            frames
            if isinstance(frames, list)
            else [frames]
            if frames is not None
            else []
        )
        self.bboxes = (
            bboxes
            if isinstance(bboxes, list) and bboxes and isinstance(bboxes[0], list)
            else [bboxes]
            if bboxes is not None
            else []
        )
        self.classes = (
            classes
            if isinstance(classes, list)
            else [classes]
            if classes is not None
            else []
        )
        self.features = feats if feats is not None else []

    def append(self, frame: int, score: float, bbox: list[float], cls: int, feat: np.ndarray) -> None:
        """Appends a detection with its embedding (keeps all lists in sync)."""
        self.times.append(frame)
        self.scores.append(score)
        self.bboxes.append(bbox)
        self.classes.append(cls)
        self.features.append(feat)

    def append_det(self, frame: int, score: float, bbox: list[float]) -> None:
        """Appends a detection to the tracklet (legacy, no feature sync)."""
        self.scores.append(score)
        self.times.append(frame)
        self.bboxes.append(bbox)

    def append_feat(self, feat: np.ndarray) -> None:
        """Appends an L2-normalised feature vector."""
        self.features.append(feat)

    def extract(self, start: int, end: int) -> "Tracklet":
        """Extracts a subtrack from index ``start`` to ``end`` (inclusive).

        Returns:
            A new Tracklet that is a subset of the original.
        """
        subtrack = Tracklet(
            self.track_id,
            self.times[start : end + 1],
            self.scores[start : end + 1],
            self.bboxes[start : end + 1],
            self.features[start : end + 1] if self.features else None,
            self.classes[start : end + 1] if self.classes else None,
        )
        subtrack.parent_id = self.track_id
        return subtrack

    def sort_by_time(self) -> None:
        """Sort all parallel lists by ascending frame time (in-place)."""
        if not self.times:
            return
        sort_idx = sorted(range(len(self.times)), key=lambda k: self.times[k])
        self.times = [self.times[k] for k in sort_idx]
        self.bboxes = [self.bboxes[k] for k in sort_idx]
        self.scores = [self.scores[k] for k in sort_idx]
        if self.classes:
            self.classes = [self.classes[k] for k in sort_idx]
        if self.features:
            self.features = [self.features[k] for k in sort_idx]

    def merge_from(self, other: "Tracklet") -> None:
        """Merge another tracklet into this one, maintaining time order."""
        self.features += other.features
        self.times += other.times
        self.bboxes += other.bboxes
        self.scores += other.scores
        self.classes += other.classes
        self.sort_by_time()


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def find_consecutive_segments(track_times: list[int]) -> list[tuple[int, int]]:
    """Identifies start and end indices of consecutive frame segments.

    Args:
        track_times: Sorted list of frame numbers.

    Returns:
        List of (start_index, end_index) tuples for each consecutive run.
    """
    if not track_times:
        return []
    segments = []
    start_index = 0
    end_index = 0
    for i in range(1, len(track_times)):
        if track_times[i] == track_times[end_index] + 1:
            end_index = i
        else:
            segments.append((start_index, end_index))
            start_index = i
            end_index = i
    segments.append((start_index, end_index))
    return segments


def query_subtracks(
    seg1: list[tuple[int, int]],
    seg2: list[tuple[int, int]],
    track1: Tracklet,
    track2: Tracklet,
) -> list[Tracklet]:
    """Pairs segments from two tracks into temporally-sorted subtracks.

    Processes non-overlapping segments from two tracks and returns them
    sorted by their starting frame time.

    Args:
        seg1: Segments from track1 as (start_index, end_index) tuples.
        seg2: Segments from track2 as (start_index, end_index) tuples.
        track1: First tracklet.
        track2: Second tracklet.

    Returns:
        List of subtracks sorted ascending by time.
    """
    # Make copies to avoid mutating the caller's lists
    seg1 = list(seg1)
    seg2 = list(seg2)

    subtracks: list[Tracklet] = []
    while seg1 and seg2:
        s1_start, s1_end = seg1[0]
        s2_start, s2_end = seg2[0]

        subtrack_1 = track1.extract(s1_start, s1_end)
        subtrack_2 = track2.extract(s2_start, s2_end)

        s1_start_frame = track1.times[s1_start]
        s2_start_frame = track2.times[s2_start]

        if s1_start_frame < s2_start_frame:
            subtracks.append(subtrack_1)
            subtracks.append(subtrack_2)
        else:
            subtracks.append(subtrack_2)
            subtracks.append(subtrack_1)
        seg1.pop(0)
        seg2.pop(0)

    # Handle remaining segments
    seg_remain = seg1 if seg1 else seg2
    track_remain = track1 if seg1 else track2
    for s_start, s_end in seg_remain:
        subtracks.append(track_remain.extract(s_start, s_end))

    return subtracks


# ---------------------------------------------------------------------------
# Distance computation
# ---------------------------------------------------------------------------


def get_distance(track1: Tracklet, track2: Tracklet) -> float:
    """Computes average pairwise cosine distance between two tracklets (Eq. 1).

    If the tracks have temporal overlap, returns 1.0 (maximum distance).
    Features are assumed to be L2-normalised.

    Returns:
        float: Average cosine distance in [0, 1].
    """
    if track1.track_id == track2.track_id:
        return 0.0

    # Temporal overlap check
    if set(track1.times) & set(track2.times):
        return 1.0

    # Features should already be L2-normalised from generation step.
    # Compute cosine distance directly: 1 - (a . b) for unit vectors.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feats_a = torch.tensor(
        np.stack(track1.features), dtype=torch.float32, device=device
    )
    feats_b = torch.tensor(
        np.stack(track2.features), dtype=torch.float32, device=device
    )

    # Normalise (defensive, in case features aren't perfectly unit-norm)
    feats_a = feats_a / feats_a.norm(dim=1, keepdim=True).clamp(min=1e-8)
    feats_b = feats_b / feats_b.norm(dim=1, keepdim=True).clamp(min=1e-8)

    cos_sim = feats_a @ feats_b.T  # (N_a, N_b)
    cos_dist = 1.0 - cos_sim

    n_a, n_b = cos_dist.shape
    avg_dist = cos_dist.sum().item() / (n_a * n_b)
    return avg_dist


def get_distance_matrix(tid2track: dict[int, Tracklet]) -> np.ndarray:
    """Builds a symmetric pairwise distance matrix for all tracklets.

    Args:
        tid2track: Mapping of track_id -> Tracklet.

    Returns:
        Square numpy array of shape (N, N) with pairwise cosine distances.
    """
    track_list = list(tid2track.values())
    n = len(track_list)
    dist = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            d = get_distance(track_list[i], track_list[j])
            dist[i, j] = d
            dist[j, i] = d

    return dist


# ---------------------------------------------------------------------------
# Spatial constraints (paper Eq. 3-4)
# ---------------------------------------------------------------------------


def get_spatial_constraints(
    tid2track: dict[int, Tracklet], factor: float
) -> tuple[float, float]:
    """Calculates spatial constraint gates from bounding box extents.

    Args:
        tid2track: Mapping of track_id -> Tracklet.
        factor: Scaling factor beta for the ranges.

    Returns:
        Tuple of (max_x_range, max_y_range) scaled by factor.
    """
    min_x = float("inf")
    max_x = float("-inf")
    min_y = float("inf")
    max_y = float("-inf")

    for track in tid2track.values():
        for bbox in track.bboxes:
            x, y, w, h = bbox[0:4]
            cx = x + w / 2
            cy = y + h / 2
            min_x = min(min_x, cx)
            max_x = max(max_x, cx)
            min_y = min(min_y, cy)
            max_y = max(max_y, cy)

    x_range = abs(max_x - min_x) * factor
    y_range = abs(max_y - min_y) * factor

    return x_range, y_range


def check_spatial_constraints(
    trk_1: Tracklet,
    trk_2: Tracklet,
    max_x_range: float,
    max_y_range: float,
) -> bool:
    """Checks if two tracklets satisfy spatial constraints for merging.

    Verifies that at every transition point between the two tracks, the exit
    location of one is within (max_x_range, max_y_range) of the entry
    location of the other.

    Returns:
        True if spatial constraints are satisfied.
    """
    seg_1 = find_consecutive_segments(trk_1.times)
    seg_2 = find_consecutive_segments(trk_2.times)

    subtracks = query_subtracks(seg_1, seg_2, trk_1, trk_2)
    if len(subtracks) < 2:
        return True

    subtrack_1st = subtracks[0]
    for subtrack_2nd in subtracks[1:]:
        if subtrack_1st.parent_id == subtrack_2nd.parent_id:
            subtrack_1st = subtrack_2nd
            continue

        x_1, y_1, w_1, h_1 = subtrack_1st.bboxes[-1][0:4]
        x_2, y_2, w_2, h_2 = subtrack_2nd.bboxes[0][0:4]
        cx_1 = x_1 + w_1 / 2
        cy_1 = y_1 + h_1 / 2
        cx_2 = x_2 + w_2 / 2
        cy_2 = y_2 + h_2 / 2
        dx = abs(cx_1 - cx_2)
        dy = abs(cy_1 - cy_2)

        if dx > max_x_range or dy > max_y_range:
            return False

        subtrack_1st = subtrack_2nd

    return True


# ---------------------------------------------------------------------------
# Splitter (Section 3.2 of paper)
# ---------------------------------------------------------------------------


def detect_id_switch(
    embs: np.ndarray,
    eps: float = 0.7,
    min_samples: int = 10,
    max_clusters: Optional[int] = None,
) -> tuple[bool, np.ndarray]:
    """Detects identity switches within a tracklet using DBSCAN clustering.

    Args:
        embs: Stacked embedding array of shape (N, D).
        eps: DBSCAN neighbourhood radius (cosine distance).
        min_samples: DBSCAN min_samples.
        max_clusters: If set, merges clusters down to this limit.

    Returns:
        Tuple of (id_switch_detected, cluster_labels).
    """
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler

    if len(embs) > 15000:
        embs = embs[::2]

    embs = np.asarray(embs)

    scaler = StandardScaler()
    embs_scaled = scaler.fit_transform(embs)

    db = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(embs_scaled)
    labels = db.labels_.copy()

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]

    # Reassign noise points to nearest cluster
    if -1 in labels and len(unique_labels) > 0:
        cluster_centers = np.array(
            [embs_scaled[labels == lbl].mean(axis=0) for lbl in unique_labels]
        )
        noise_indices = np.where(labels == -1)[0]
        for idx in noise_indices:
            distances = cdist([embs_scaled[idx]], cluster_centers, metric="cosine")
            nearest = np.argmin(distances)
            labels[idx] = unique_labels[nearest]

    # Recount after noise reassignment
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]
    n_clusters = len(unique_labels)

    # Merge excess clusters
    if max_clusters and n_clusters > max_clusters:
        while n_clusters > max_clusters:
            cluster_centers = np.array(
                [embs_scaled[labels == lbl].mean(axis=0) for lbl in unique_labels]
            )
            distance_matrix = cdist(
                cluster_centers, cluster_centers, metric="cosine"
            )
            np.fill_diagonal(distance_matrix, np.inf)

            min_dist_idx = np.unravel_index(
                np.argmin(distance_matrix), distance_matrix.shape
            )
            merge_from = unique_labels[min_dist_idx[1]]
            merge_to = unique_labels[min_dist_idx[0]]
            labels[labels == merge_from] = merge_to

            unique_labels = np.unique(labels)
            unique_labels = unique_labels[unique_labels != -1]
            n_clusters = len(unique_labels)

    return n_clusters > 1, labels


def split_tracklets(
    tmp_trklets: dict[int, Tracklet],
    eps: float = 0.7,
    max_k: int = 3,
    min_samples: int = 10,
    len_thres: int = 100,
) -> dict[int, Tracklet]:
    """Splits tracklets that contain multiple identities.

    Uses DBSCAN clustering on ReID embeddings to detect identity switches
    and splits affected tracklets into separate pure-identity tracklets.

    Args:
        tmp_trklets: Input tracklets dict (track_id -> Tracklet).
        eps: DBSCAN eps parameter (cosine distance).
        max_k: Maximum number of output clusters per tracklet.
        min_samples: DBSCAN min_samples parameter.
        len_thres: Minimum tracklet length to consider for splitting.

    Returns:
        New dict of tracklets after splitting.
    """
    new_id = max(tmp_trklets.keys()) + 1
    tracklets: dict[int, Tracklet] = {}

    for tid in tqdm(
        sorted(tmp_trklets.keys()),
        total=len(tmp_trklets),
        desc="Splitting tracklets",
    ):
        trklet = tmp_trklets[tid]
        if len(trklet.times) < len_thres:
            tracklets[tid] = trklet
        else:
            embs = np.stack(trklet.features)
            frames = np.array(trklet.times)
            bboxes = np.stack(trklet.bboxes)
            scores = np.array(trklet.scores)
            classes = np.array(trklet.classes)

            id_switch_detected, clusters = detect_id_switch(
                embs, eps=eps, min_samples=min_samples, max_clusters=max_k
            )

            if not id_switch_detected:
                tracklets[tid] = trklet
            else:
                unique_labels = set(clusters)
                for label in unique_labels:
                    if label == -1:
                        continue
                    mask = clusters == label
                    tmp_embs = embs[mask]
                    tmp_frames = frames[mask]
                    tmp_bboxes = bboxes[mask]
                    tmp_scores = scores[mask]
                    tmp_classes = classes[mask]

                    tracklets[new_id] = Tracklet(
                        new_id,
                        tmp_frames.tolist(),
                        tmp_scores.tolist(),
                        tmp_bboxes.tolist(),
                        feats=[e for e in tmp_embs],
                        classes=tmp_classes.tolist(),
                    )
                    new_id += 1

    return tracklets


# ---------------------------------------------------------------------------
# Connector / Merger (Section 3.3 of paper)
# ---------------------------------------------------------------------------


def merge_tracklets(
    tracklets: dict[int, Tracklet],
    merge_dist_thres: float,
    max_x_range: float,
    max_y_range: float,
) -> dict[int, Tracklet]:
    """Hierarchical agglomerative merging of tracklets.

    Repeatedly merges the closest pair of tracklets (by average pairwise
    cosine distance) until no pair is below the threshold, subject to
    temporal non-overlap and spatial constraints.

    Args:
        tracklets: Dict of track_id -> Tracklet.
        merge_dist_thres: Maximum cosine distance for merging.
        max_x_range: Spatial gate in x.
        max_y_range: Spatial gate in y.

    Returns:
        Merged tracklets dict.
    """
    if len(tracklets) <= 1:
        return tracklets

    # Build initial distance matrix
    tid_list = list(tracklets.keys())
    n = len(tid_list)
    dist = np.ones((n, n), dtype=np.float64)
    np.fill_diagonal(dist, np.inf)

    for i in range(n):
        for j in range(i + 1, n):
            d = get_distance(tracklets[tid_list[i]], tracklets[tid_list[j]])
            dist[i, j] = d
            dist[j, i] = d

    while True:
        min_val = dist.min()
        if min_val >= merge_dist_thres:
            break

        # Find minimum distance pair
        min_idx = np.unravel_index(np.argmin(dist), dist.shape)
        idx_a, idx_b = min_idx[0], min_idx[1]

        track_a = tracklets[tid_list[idx_a]]
        track_b = tracklets[tid_list[idx_b]]

        # Temporal overlap check (defensive - should already be dist=1)
        if set(track_a.times) & set(track_b.times):
            dist[idx_a, idx_b] = merge_dist_thres
            dist[idx_b, idx_a] = merge_dist_thres
            continue

        # Spatial constraint check
        if not check_spatial_constraints(
            track_a, track_b, max_x_range, max_y_range
        ):
            dist[idx_a, idx_b] = merge_dist_thres
            dist[idx_b, idx_a] = merge_dist_thres
            continue

        # Merge track_b into track_a (includes scores + sorts by time)
        track_a.merge_from(track_b)

        # Remove track_b from data structures
        del tracklets[tid_list[idx_b]]

        # Delete row/column for track_b from distance matrix
        dist = np.delete(dist, idx_b, axis=0)
        dist = np.delete(dist, idx_b, axis=1)

        # Fix: adjust idx_a if idx_b was before it
        if idx_b < idx_a:
            idx_a -= 1

        # Rebuild tid_list after removal
        tid_list = list(tracklets.keys())

        # Update only the merged track's row/column
        for k in range(dist.shape[0]):
            if k == idx_a:
                dist[k, k] = np.inf
            else:
                d = get_distance(
                    tracklets[tid_list[idx_a]], tracklets[tid_list[k]]
                )
                dist[idx_a, k] = d
                dist[k, idx_a] = d

    return tracklets


def merge_tracklets_batched(
    tracklets: dict[int, Tracklet],
    batch_size: int = 50,
    max_x_range: float = 0.0,
    max_y_range: float = 0.0,
    merge_dist_thres: float = 0.4,
) -> dict[int, Tracklet]:
    """Batched hierarchical merging for large tracklet sets.

    First merges within batches, then performs a global merge pass
    to catch cross-batch associations.

    Args:
        tracklets: Dict of track_id -> Tracklet.
        batch_size: Number of tracklets per batch.
        max_x_range: Spatial gate in x.
        max_y_range: Spatial gate in y.
        merge_dist_thres: Cosine distance threshold for merging.

    Returns:
        Merged tracklets dict.
    """
    tracklet_items = list(tracklets.items())
    temp_tracklets: dict[int, Tracklet] = {}

    LOGGER.info(
        f"Batched merge: {len(tracklet_items)} tracklets, "
        f"batch_size={batch_size}"
    )

    for i in range(0, len(tracklet_items), batch_size):
        batch = dict(tracklet_items[i : i + batch_size])
        merged_batch = merge_tracklets(
            batch, merge_dist_thres, max_x_range, max_y_range
        )
        LOGGER.debug(
            f"Batch [{i}:{i + len(batch)}]: "
            f"{len(batch)} -> {len(merged_batch)} tracklets"
        )
        temp_tracklets.update(merged_batch)

    # Global merge pass across all batches
    LOGGER.info(f"Global merge pass: {len(temp_tracklets)} tracklets")
    merged = merge_tracklets(
        temp_tracklets, merge_dist_thres, max_x_range, max_y_range
    )
    return merged


# Public names are limited to the pure, in-memory algorithms. GTA's former
# model-owning CLI and positional-cache adapter are intentionally absent in v24.
__all__ = (
    "Tracklet",
    "check_spatial_constraints",
    "detect_id_switch",
    "find_consecutive_segments",
    "get_distance",
    "get_distance_matrix",
    "get_spatial_constraints",
    "merge_tracklets",
    "merge_tracklets_batched",
    "query_subtracks",
    "split_tracklets",
)
