# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

"""Offline Global Tracklet Association (GTA) module.

Implements the postprocessing pipeline from:
    Sun et al., "GTA: Global Tracklet Association for Multi-Object Tracking
    in Sports", ACCV 2024 Workshop.

The pipeline operates on complete tracklets after tracking and consists of:

1. **Tracklet Generation**: Extracts ReID features from tracking results and
   original video frames, producing per-sequence tracklet pickles.

2. **Tracklet Splitter**: Detects identity switches within a single tracklet
   using DBSCAN clustering on ReID embeddings and splits mixed-identity
   tracklets into separate pure-identity tracklets.

3. **Tracklet Connector**: Merges tracklets belonging to the same identity
   using hierarchical agglomerative clustering with average pairwise cosine
   distance (Eq. 1 from the paper), subject to spatial constraints.

Usage
-----
    # Generate tracklets
    python -m boxmot.postprocessing.gta generate \
        --model_path reid_model.pth \
        --data_path /path/to/MOT/sequences \
        --pred_dir /path/to/tracker/results \
        --tracker MyTracker

    # Run split + connect postprocessing
    python -m boxmot.postprocessing.gta associate \
        --dataset SportsMOT \
        --tracker MyTracker \
        --track_src /path/to/Tracklets_dir \
        --use_split --use_connect
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from scipy.spatial.distance import cdist

from boxmot.postprocessing.base import Postprocessor, ProgressCallback
from boxmot.utils import logger as LOGGER
from boxmot.utils.callbacks import safe_seq_progress_callback
from boxmot.utils.rich.progress import RichTqdm as tqdm

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


# ---------------------------------------------------------------------------
# I/O: Tracklet generation
# ---------------------------------------------------------------------------


def generate_tracklets(
    model_path: str,
    data_path: str,
    pred_dir: str,
    tracker: str,
) -> None:
    """Generate tracklets with ReID features from tracking results.

    Reads tracking results in MOT format, extracts per-detection crops,
    runs them through a ReID model, and saves per-sequence tracklet pickles.

    Args:
        model_path: Path to the ReID model weights.
        data_path: Path to the dataset split (containing sequence folders).
        pred_dir: Directory with tracking result .txt files.
        tracker: Tracker name (used for output directory naming).
    """
    from torchreid.utils import FeatureExtractor

    val_transforms = T.Compose(
        [
            T.Resize([256, 128]),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    split = os.path.basename(data_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = FeatureExtractor(
        model_name="osnet_x1_0",
        model_path=model_path,
        device=device,
    )

    output_dir = os.path.join(
        os.path.dirname(pred_dir), f"{tracker}_Tracklets_{split}"
    )
    os.makedirs(output_dir, exist_ok=True)

    seqs = sorted([f for f in os.listdir(pred_dir) if f.endswith(".txt")])

    for seq_file in tqdm(seqs, desc="Processing sequences"):
        seq = seq_file.replace(".txt", "")
        imgs = sorted(
            glob.glob(os.path.join(data_path, seq, "img1", "*"))
        )
        track_res = np.genfromtxt(
            os.path.join(pred_dir, seq_file), dtype=float, delimiter=","
        )

        if track_res.ndim == 1:
            track_res = track_res.reshape(1, -1)

        last_frame = int(track_res[-1][0])
        seq_tracks: dict[int, Tracklet] = {}

        for frame_id in range(1, last_frame + 1):
            inds = track_res[:, 0] == frame_id
            frame_res = track_res[inds]
            if len(frame_res) == 0:
                continue

            img = Image.open(imgs[int(frame_id) - 1])
            input_batch = None
            tid2idx: dict[int, int] = {}

            # MOT format: <frame>, <id>, <left>, <top>, <w>, <h>, <conf>, ...
            for idx, row in enumerate(frame_res):
                frame, track_id = int(row[0]), int(row[1])
                left, top, w, h = row[2], row[3], row[4], row[5]
                score = row[6] if len(row) > 6 else 1.0

                bbox = [left, top, w, h]
                if track_id not in seq_tracks:
                    seq_tracks[track_id] = Tracklet(
                        track_id, frame, score, bbox
                    )
                else:
                    seq_tracks[track_id].append_det(frame, score, bbox)
                tid2idx[track_id] = idx

                im = img.crop(
                    (left, top, left + w, top + h)
                ).convert("RGB")
                im_tensor = val_transforms(im).unsqueeze(0)
                if input_batch is None:
                    input_batch = im_tensor
                else:
                    input_batch = torch.cat(
                        [input_batch, im_tensor], dim=0
                    )

            if input_batch is not None:
                features = extractor(input_batch)
                feats = features.cpu().detach().numpy()

                for tid, idx in tid2idx.items():
                    feat = feats[idx]
                    feat = feat / (np.linalg.norm(feat) + 1e-8)
                    seq_tracks[tid].append_feat(feat)

        track_output_path = os.path.join(output_dir, f"{seq}.pkl")
        with open(track_output_path, "wb") as f:
            pickle.dump(seq_tracks, f)
        LOGGER.info(f"Saved tracklets to {track_output_path}")


# ---------------------------------------------------------------------------
# I/O: Save results in MOT format
# ---------------------------------------------------------------------------


def save_results(output_path: str, tracklets: dict[int, Tracklet]) -> None:
    """Saves merged tracklets in boxmot's 9-column MOT format.

    Format: frame,id,left,top,width,height,conf,cls,det_ind
    Reassigns sequential track IDs starting from 1.

    Args:
        output_path: Path to the output .txt file.
        tracklets: Dict of track_id -> Tracklet.
    """
    results = []
    for new_tid, old_tid in enumerate(sorted(tracklets.keys()), start=1):
        track = tracklets[old_tid]
        for i, frame_id in enumerate(track.times):
            bbox = track.bboxes[i]
            conf = track.scores[i] if i < len(track.scores) else 1.0
            cls = track.classes[i] if i < len(track.classes) else 1
            results.append(
                [frame_id, new_tid, bbox[0], bbox[1], bbox[2], bbox[3], conf, cls, -1]
            )

    results.sort(key=lambda x: (x[0], x[1]))

    with open(output_path, "w") as f:
        for line in results:
            f.write(
                f"{int(line[0])},{int(line[1])},"
                f"{int(round(line[2]))},{int(round(line[3]))},"
                f"{int(round(line[4]))},{int(round(line[5]))},"
                f"{line[6]:.6f},{int(line[7])},{int(line[8])}\n"
            )
    LOGGER.info(f"Saved results to {output_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_generate_args(
    subparsers: argparse._SubParsersAction,
) -> None:
    """Add 'generate' subcommand."""
    p = subparsers.add_parser(
        "generate", help="Generate tracklets with ReID features."
    )
    p.add_argument(
        "--model_path", type=str, required=True, help="Path to ReID model."
    )
    p.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to dataset split directory.",
    )
    p.add_argument(
        "--pred_dir",
        type=str,
        required=True,
        help="Directory with tracker .txt results.",
    )
    p.add_argument(
        "--tracker", type=str, required=True, help="Tracker name."
    )


def parse_associate_args(
    subparsers: argparse._SubParsersAction,
) -> None:
    """Add 'associate' subcommand."""
    p = subparsers.add_parser(
        "associate", help="Run GTA split + connect postprocessing."
    )
    p.add_argument(
        "--dataset", type=str, required=True, help="Dataset name."
    )
    p.add_argument(
        "--tracker", type=str, required=True, help="Tracker name."
    )
    p.add_argument(
        "--track_src",
        type=str,
        required=True,
        help="Source directory of tracklet .pkl files.",
    )
    p.add_argument(
        "--use_split", action="store_true", help="Enable split component."
    )
    p.add_argument(
        "--use_connect",
        action="store_true",
        help="Enable connect component.",
    )
    p.add_argument(
        "--min_len",
        type=int,
        default=100,
        help="Minimum tracklet length for splitting.",
    )
    p.add_argument(
        "--eps",
        type=float,
        default=0.7,
        help="DBSCAN eps (cosine distance).",
    )
    p.add_argument(
        "--min_samples",
        type=int,
        default=10,
        help="DBSCAN min_samples.",
    )
    p.add_argument(
        "--max_k",
        type=int,
        default=3,
        help="Maximum clusters from splitting.",
    )
    p.add_argument(
        "--spatial_factor",
        type=float,
        default=1.0,
        help="Spatial constraint factor beta.",
    )
    p.add_argument(
        "--merge_dist_thres",
        type=float,
        default=0.4,
        help="Cosine distance threshold for merging.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Batch size for batched merging.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GTA: Global Tracklet Association postprocessing."
    )
    subparsers = parser.add_subparsers(dest="command")
    parse_generate_args(subparsers)
    parse_associate_args(subparsers)

    args = parser.parse_args()

    if args.command == "generate":
        generate_tracklets(
            args.model_path, args.data_path, args.pred_dir, args.tracker
        )

    elif args.command == "associate":
        if not args.use_split and not args.use_connect:
            parser.error(
                "At least one of --use_split or --use_connect is required."
            )

        if args.use_split and args.use_connect:
            process = "Split+Connect"
        elif args.use_split:
            process = "Split"
        else:
            process = "Connect"

        seq_tracks_dir = args.track_src
        data_path = os.path.dirname(seq_tracks_dir)
        seqs_tracks = sorted(os.listdir(seq_tracks_dir))

        for seq_idx, seq in enumerate(seqs_tracks):
            seq_name = seq.split(".")[0]
            LOGGER.info(
                f"Processing sequence {seq_idx + 1}/{len(seqs_tracks)}: "
                f"{seq_name}"
            )

            with open(
                os.path.join(seq_tracks_dir, seq), "rb"
            ) as pkl_f:
                tmp_trklets: dict[int, Tracklet] = pickle.load(pkl_f)

            max_x_range, max_y_range = get_spatial_constraints(
                tmp_trklets, args.spatial_factor
            )

            # --- Split phase ---
            if args.use_split:
                LOGGER.info(
                    f"Tracklets before splitting: {len(tmp_trklets)}"
                )
                split_trklets = split_tracklets(
                    tmp_trklets,
                    eps=args.eps,
                    max_k=args.max_k,
                    min_samples=args.min_samples,
                    len_thres=args.min_len,
                )
            else:
                split_trklets = tmp_trklets

            # --- Connect/merge phase ---
            if args.use_connect:
                LOGGER.info(
                    f"Tracklets before merging: {len(split_trklets)}"
                )
                merged_trklets = merge_tracklets_batched(
                    split_trklets,
                    batch_size=args.batch_size,
                    max_x_range=max_x_range,
                    max_y_range=max_y_range,
                    merge_dist_thres=args.merge_dist_thres,
                )
                LOGGER.info(
                    f"Tracklets after merging: {len(merged_trklets)}"
                )
            else:
                merged_trklets = split_trklets

            # --- Save ---
            sct_name = (
                f"{args.tracker}_{args.dataset}_{process}"
                f"_eps{args.eps}_minSamples{args.min_samples}"
                f"_K{args.max_k}_mergeDist{args.merge_dist_thres}"
                f"_spatial{args.spatial_factor}"
            )
            os.makedirs(
                os.path.join(data_path, sct_name), exist_ok=True
            )
            output_path = os.path.join(
                data_path, sct_name, f"{seq_name}.txt"
            )
            save_results(output_path, merged_trklets)

    else:
        parser.print_help()


# ---------------------------------------------------------------------------
# Eval-pipeline integration: postprocess from cached embeddings
# ---------------------------------------------------------------------------


class GTAPostprocessor(Postprocessor):
    """Global tracklet association postprocessor."""

    name = "gta"
    display_name = "GTA"

    def __init__(
        self,
        embs_dir: Path | None = None,
        dets_dir: Path | None = None,
        use_split: bool = True,
        use_connect: bool = True,
        eps: float = 0.7,
        min_samples: int = 10,
        max_k: int = 3,
        min_len: int = 100,
        merge_dist_thres: float = 0.4,
        spatial_factor: float = 1.0,
        batch_size: int = 50,
    ) -> None:
        self.embs_dir = embs_dir
        self.dets_dir = dets_dir
        self.use_split = use_split
        self.use_connect = use_connect
        self.eps = eps
        self.min_samples = min_samples
        self.max_k = max_k
        self.min_len = min_len
        self.merge_dist_thres = merge_dist_thres
        self.spatial_factor = spatial_factor
        self.batch_size = batch_size

    def run(
        self,
        mot_results_folder: Path,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        """Run GTA postprocessing on MOT result files using cached embeddings."""
        embs_dir = self.embs_dir
        dets_dir = self.dets_dir
        use_split = self.use_split
        use_connect = self.use_connect
        eps = self.eps
        min_samples = self.min_samples
        max_k = self.max_k
        min_len = self.min_len
        merge_dist_thres = self.merge_dist_thres
        spatial_factor = self.spatial_factor
        batch_size = self.batch_size

        mot_results_folder = Path(mot_results_folder)
        result_files = sorted(mot_results_folder.glob("*.txt"))

        if not result_files:
            LOGGER.warning(f"GTA: No .txt files found in {mot_results_folder}")
            return

        if embs_dir is None:
            LOGGER.warning(
                "GTA: No embeddings directory provided, skipping postprocessing. "
                "GTA requires appearance embeddings from a ReID model."
            )
            return

        embs_dir = Path(embs_dir)
        dets_dir = Path(dets_dir) if dets_dir is not None else None

        progress_callback = safe_seq_progress_callback(progress_callback)
        total_files = len(result_files)
        for file_idx, result_file in enumerate(result_files, 1):
            seq_name = result_file.stem
            LOGGER.info(f"GTA postprocessing: {seq_name}")

            # Load MOT results: [frame, id, l, t, w, h, conf, cls, det_ind]
            try:
                mot_data = np.loadtxt(result_file, delimiter=",")
            except (ValueError, OSError) as exc:
                LOGGER.warning(f"GTA: could not load {result_file}: {exc}. Skipping {seq_name}...")
                continue
            if mot_data.size == 0:
                continue
            if mot_data.ndim == 1:
                mot_data = mot_data.reshape(1, -1)

            # Load cached embeddings for this sequence
            emb_path = embs_dir / f"{seq_name}.npy"
            if not emb_path.exists():
                LOGGER.warning(f"GTA: Embedding file not found: {emb_path}, skipping {seq_name}")
                continue
            all_embs = np.load(emb_path, mmap_mode="r")

            # Load cached detections (needed to map frame_id -> row indices)
            det_data = None
            if dets_dir is not None:
                det_path = dets_dir / f"{seq_name}.npy"
                if det_path.exists():
                    det_data = np.load(det_path, mmap_mode="r")

            # Build Tracklet objects from MOT results + embeddings
            tracklets = _build_tracklets_from_mot(mot_data, all_embs, det_data)
            if not tracklets:
                continue

            # Compute spatial constraints
            max_x_range, max_y_range = get_spatial_constraints(tracklets, spatial_factor)

            # Split phase
            if use_split:
                LOGGER.debug(f"  Tracklets before split: {len(tracklets)}")
                tracklets = split_tracklets(
                    tracklets, eps=eps, max_k=max_k,
                    min_samples=min_samples, len_thres=min_len,
                )

            # Connect/merge phase
            if use_connect:
                LOGGER.debug(f"  Tracklets before merge: {len(tracklets)}")
                tracklets = merge_tracklets_batched(
                    tracklets,
                    batch_size=batch_size,
                    max_x_range=max_x_range,
                    max_y_range=max_y_range,
                    merge_dist_thres=merge_dist_thres,
                )
                LOGGER.debug(f"  Tracklets after merge: {len(tracklets)}")

            # Overwrite the MOT result file
            save_results(str(result_file), tracklets)
            if progress_callback is not None:
                progress_callback(seq_name, file_idx, total_files)


def gta(
    mot_results_folder: Path,
    embs_dir: Path | None = None,
    dets_dir: Path | None = None,
    use_split: bool = True,
    use_connect: bool = True,
    eps: float = 0.7,
    min_samples: int = 10,
    max_k: int = 3,
    min_len: int = 100,
    merge_dist_thres: float = 0.4,
    spatial_factor: float = 1.0,
    batch_size: int = 50,
    progress_callback: ProgressCallback | None = None,
) -> None:
    """Run GTA postprocessing on MOT result files using cached embeddings.

    Designed to be called from the eval pipeline after tracking has completed.
    Reads MOT result .txt files from ``mot_results_folder``, loads corresponding
    cached embeddings via the ``det_ind`` column, builds Tracklet objects, and
    runs the split + connect pipeline in-place.
    """
    GTAPostprocessor(
        embs_dir=embs_dir,
        dets_dir=dets_dir,
        use_split=use_split,
        use_connect=use_connect,
        eps=eps,
        min_samples=min_samples,
        max_k=max_k,
        min_len=min_len,
        merge_dist_thres=merge_dist_thres,
        spatial_factor=spatial_factor,
        batch_size=batch_size,
    ).run(mot_results_folder, progress_callback=progress_callback)


def _build_tracklets_from_mot(
    mot_data: np.ndarray,
    all_embs: np.ndarray,
    det_data: np.ndarray | None = None,
) -> dict[int, Tracklet]:
    """Build Tracklet objects from MOT result rows + cached embeddings.

    Only detections with a resolvable embedding are included, which ensures
    len(features) == len(times) == len(bboxes) == len(scores) == len(classes).

    Args:
        mot_data: MOT results, shape (N, >=8). Columns:
            [frame, track_id, left, top, w, h, conf, cls, (det_ind)].
        all_embs: All cached embeddings for the sequence, shape (M, D).
        det_data: Cached detections for the sequence, shape (M, cols).
            First column is frame_id (used for frame->row mapping).

    Returns:
        Dict of track_id -> Tracklet with embeddings attached.
    """
    tracklets: dict[int, Tracklet] = {}

    # Determine if det_ind column is available (column 8, 0-indexed)
    has_det_ind = mot_data.shape[1] > 8

    # Build frame_id -> base row index mapping from det_data
    frame_to_base_idx: dict[int, int] = {}
    if det_data is not None:
        # det_data first column is frame_id
        frame_ids_det = det_data[:, 0].astype(int)
        current_frame = -1
        for row_idx, fid in enumerate(frame_ids_det):
            if fid != current_frame:
                frame_to_base_idx[fid] = row_idx
                current_frame = fid

    skipped = 0

    # Process each row in MOT data
    for row in mot_data:
        frame_id = int(row[0])
        track_id = int(row[1])
        left, top, w, h = row[2], row[3], row[4], row[5]
        conf = float(row[6]) if mot_data.shape[1] > 6 else 1.0
        cls = int(row[7]) if mot_data.shape[1] > 7 else 1
        bbox = [left, top, w, h]

        # Resolve embedding index
        emb_idx = None
        if has_det_ind:
            det_ind = int(row[8])
            if det_ind >= 0:
                # det_ind is frame-local; convert to global row index
                base = frame_to_base_idx.get(frame_id, None)
                if base is not None:
                    emb_idx = base + det_ind

        # Skip detections without a valid embedding (keeps lists in sync)
        if emb_idx is None or emb_idx < 0 or emb_idx >= len(all_embs):
            skipped += 1
            continue

        feat = all_embs[emb_idx].astype(np.float32)
        norm = np.linalg.norm(feat)
        if norm < 1e-8:
            skipped += 1
            continue
        feat = feat / norm

        # Build/extend tracklet (all lists grow together)
        if track_id not in tracklets:
            tracklets[track_id] = Tracklet(track_id)
        tracklets[track_id].append(frame_id, conf, bbox, cls, feat)

    if skipped > 0:
        total = len(mot_data)
        LOGGER.debug(
            f"  _build_tracklets_from_mot: skipped {skipped}/{total} "
            f"detections without valid embeddings"
        )

    # Remove tracklets that ended up empty
    tracklets = {
        tid: trk for tid, trk in tracklets.items()
        if len(trk.features) > 0
    }

    return tracklets


if __name__ == "__main__":
    main()
