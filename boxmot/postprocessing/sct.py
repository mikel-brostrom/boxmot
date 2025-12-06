"""Split-and-Connect Tracklets (SCT) postprocessing.

This module ports a research prototype for splitting erroneous identity
switches inside a track and reconnecting fragmented trajectories based on
appearance similarity. It mirrors :mod:`boxmot.postprocessing.gsi` in how it
is wired into the CLI: given a folder with ``MOT*.txt`` files, it rewrites the
files in place after postprocessing.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from boxmot.utils import logger as LOGGER


@dataclass
class Tracklet:
    """Container representing a single trajectory.

    Attributes:
        track_id (int): Identifier for the tracklet.
        times (List[int]): Frame indices where the tracklet is present.
        scores (List[float]): Detection scores for each frame.
        bboxes (List[List[float]]): Bounding boxes ``[x, y, w, h]`` per frame.
        features (List[np.ndarray]): Appearance embeddings associated with the
            detections. In this port, we derive compact features from the
            bounding boxes and scores when raw embeddings are not available.
    """

    track_id: int
    times: List[int] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    bboxes: List[List[float]] = field(default_factory=list)
    features: List[np.ndarray] = field(default_factory=list)
    parent_id: Optional[int] = None

    def __post_init__(self) -> None:
        if self.parent_id is None:
            self.parent_id = self.track_id

    def append_det(self, frame: int, score: float, bbox: Sequence[float]) -> None:
        """Append a detection to the tracklet."""

        self.times.append(frame)
        self.scores.append(score)
        self.bboxes.append(list(bbox))

    def append_feat(self, feat: np.ndarray) -> None:
        """Append an appearance feature to the tracklet."""

        self.features.append(feat)

    def extract(self, start: int, end: int) -> "Tracklet":
        """Return a subtrack between two indices (inclusive)."""

        return Tracklet(
            self.track_id,
            self.times[start : end + 1],
            self.scores[start : end + 1],
            self.bboxes[start : end + 1],
            self.features[start : end + 1],
            parent_id=self.parent_id,
        )


def find_consecutive_segments(track_times: Sequence[int]) -> List[Tuple[int, int]]:
    """Identify consecutive index spans within sorted frame indices."""

    segments: List[Tuple[int, int]] = []
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
    seg1: List[Tuple[int, int]], seg2: List[Tuple[int, int]], track1: Tracklet, track2: Tracklet
) -> List[Tracklet]:
    """Pair segments from two tracks into time-ordered subtracks."""

    subtracks: List[Tracklet] = []
    while seg1 and seg2:
        s1_start, s1_end = seg1[0]
        s2_start, s2_end = seg2[0]

        subtrack_1 = track1.extract(s1_start, s1_end)
        subtrack_2 = track2.extract(s2_start, s2_end)

        s1_start_frame = track1.times[s1_start]
        s2_start_frame = track2.times[s2_start]

        if s1_start_frame < s2_start_frame:
            assert track1.times[s1_end] <= s2_start_frame
            subtracks.extend([subtrack_1, subtrack_2])
        else:
            assert s1_start_frame >= track2.times[s2_end]
            subtracks.extend([subtrack_2, subtrack_1])
        seg1.pop(0)
        seg2.pop(0)

    seg_remain = seg1 if seg1 else seg2
    track_remain = track1 if seg1 else track2
    while seg_remain:
        s_start, s_end = seg_remain[0]
        if (s_end - s_start) < 30:
            seg_remain.pop(0)
            continue
        subtracks.append(track_remain.extract(s_start, s_end))
        seg_remain.pop(0)

    return subtracks


def get_spatial_constraints(tid2track: Dict[int, Tracklet], factor: float) -> Tuple[float, float]:
    """Compute maximal spatial deltas between detections as a merge gate."""

    min_x = float("inf")
    max_x = float("-inf")
    min_y = float("inf")
    max_y = float("-inf")

    for track in tid2track.values():
        for bbox in track.bboxes:
            x, y, w, h = bbox[0:4]
            x += w / 2
            y += h / 2
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

    x_range = abs(max_x - min_x) * factor
    y_range = abs(max_y - min_y) * factor

    return x_range, y_range


def get_distance_matrix(tid2track: Dict[int, Tracklet]) -> np.ndarray:
    """Build a symmetric cosine-distance matrix across all tracklets."""

    dist = np.zeros((len(tid2track), len(tid2track)))

    for i, (track1_id, track1) in enumerate(tid2track.items()):
        for j, (track2_id, track2) in enumerate(tid2track.items()):
            if j < i:
                dist[i][j] = dist[j][i]
            else:
                dist[i][j] = get_distance(track1_id, track2_id, track1, track2)
    return dist


def get_distance(track1_id: int, track2_id: int, track1: Tracklet, track2: Tracklet) -> float:
    """Compute mean cosine distance between two tracklets' embeddings."""

    assert track1_id == track1.track_id and track2_id == track2.track_id
    does_overlap = False
    if track1_id != track2_id:
        does_overlap = bool(set(track1.times) & set(track2.times))
    if does_overlap:
        return 1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    track1_features_tensor = torch.tensor(np.stack(track1.features), dtype=torch.float32, device=device)
    track2_features_tensor = torch.tensor(np.stack(track2.features), dtype=torch.float32, device=device)
    count1 = len(track1_features_tensor)
    count2 = len(track2_features_tensor)

    cos_sim_numerator = torch.matmul(track1_features_tensor, track2_features_tensor.T)
    track1_features_dist = torch.norm(track1_features_tensor, p=2, dim=1, keepdim=True)
    track2_features_dist = torch.norm(track2_features_tensor, p=2, dim=1, keepdim=True)
    cos_sim_denominator = torch.matmul(track1_features_dist, track2_features_dist.T)
    cos_dist = 1 - cos_sim_numerator / cos_sim_denominator

    total_cos_dist = cos_dist.sum()
    return (total_cos_dist / (count1 * count2)).item()


def check_spatial_constraints(
    trk_1: Tracklet, trk_2: Tracklet, max_x_range: float, max_y_range: float
) -> bool:
    """Validate that the gap between two tracklets is spatially plausible."""

    in_spatial_range = True
    seg_1 = find_consecutive_segments(trk_1.times)
    seg_2 = find_consecutive_segments(trk_2.times)

    subtracks = query_subtracks(seg_1, seg_2, trk_1, trk_2)
    subtrack_1st = subtracks.pop(0)
    while subtracks:
        subtrack_2nd = subtracks.pop(0)
        if subtrack_1st.parent_id == subtrack_2nd.parent_id:
            subtrack_1st = subtrack_2nd
            continue
        x_1, y_1, w_1, h_1 = subtrack_1st.bboxes[-1][0:4]
        x_2, y_2, w_2, h_2 = subtrack_2nd.bboxes[0][0:4]
        x_1 += w_1 / 2
        y_1 += h_1 / 2
        x_2 += w_2 / 2
        y_2 += h_2 / 2
        dx = abs(x_1 - x_2)
        dy = abs(y_1 - y_2)

        if dx > max_x_range or dy > max_y_range:
            in_spatial_range = False
            break
        subtrack_1st = subtrack_2nd
    return in_spatial_range


def merge_tracklets(
    tracklets: Dict[int, Tracklet],
    dist: np.ndarray,
    seq_name: Optional[str] = None,
    max_x_range: Optional[float] = None,
    max_y_range: Optional[float] = None,
    merge_dist_thres: Optional[float] = None,
) -> Dict[int, Tracklet]:
    """Iteratively merge tracklets whose average cosine distance is small."""

    if merge_dist_thres is None:
        return tracklets

    idx2tid = {idx: tid for idx, tid in enumerate(tracklets.keys())}
    diagonal_mask = np.eye(dist.shape[0], dtype=bool)
    non_diagonal_mask = ~diagonal_mask
    while np.any(dist[non_diagonal_mask] < merge_dist_thres):
        min_index = np.argmin(dist[non_diagonal_mask])
        track1_idx, track2_idx = np.where(non_diagonal_mask)
        track1_idx = track1_idx[min_index]
        track2_idx = track2_idx[min_index]

        track1 = tracklets[idx2tid[track1_idx]]
        track2 = tracklets[idx2tid[track2_idx]]

        in_spatial_range = check_spatial_constraints(track1, track2, max_x_range, max_y_range)
        if in_spatial_range:
            track1.features += track2.features
            track1.times += track2.times
            track1.bboxes += track2.bboxes
            track1.scores += track2.scores

            tracklets[idx2tid[track1_idx]] = track1
            tracklets.pop(idx2tid[track2_idx])

            dist = np.delete(dist, track2_idx, axis=0)
            dist = np.delete(dist, track2_idx, axis=1)
            idx2tid = {idx: tid for idx, tid in enumerate(tracklets.keys())}

            for idx in range(dist.shape[0]):
                dist[track1_idx, idx] = get_distance(
                    idx2tid[track1_idx], idx2tid[idx], tracklets[idx2tid[track1_idx]], tracklets[idx2tid[idx]]
                )
                dist[idx, track1_idx] = dist[track1_idx, idx]

            diagonal_mask = np.eye(dist.shape[0], dtype=bool)
            non_diagonal_mask = ~diagonal_mask
        else:
            dist[track1_idx, track2_idx], dist[track2_idx, track1_idx] = merge_dist_thres, merge_dist_thres

    if seq_name:
        LOGGER.info(f"Merged tracklets for {seq_name}: {len(tracklets)} remaining")
    return tracklets


def detect_id_switch(
    embs: Sequence[np.ndarray], eps: float, min_samples: int, max_clusters: Optional[int] = None
) -> Tuple[bool, np.ndarray]:
    """Detect potential identity switches using DBSCAN clustering."""

    if len(embs) > 15_000:
        embs = embs[1::2]

    embs_stack = np.stack(embs)
    scaler = StandardScaler()
    embs_scaled = scaler.fit_transform(embs_stack)

    db = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(embs_scaled)
    labels = db.labels_

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]

    if -1 in labels and len(unique_labels) > 1:
        cluster_centers = np.array([embs_scaled[labels == label].mean(axis=0) for label in unique_labels])

        noise_indices = np.where(labels == -1)[0]
        for idx in noise_indices:
            distances = cdist([embs_scaled[idx]], cluster_centers, metric="cosine")
            nearest_cluster = np.argmin(distances)
            labels[idx] = list(unique_labels)[nearest_cluster]

    n_clusters = len(unique_labels)

    if max_clusters and n_clusters > max_clusters:
        while n_clusters > max_clusters:
            cluster_centers = np.array([embs_scaled[labels == label].mean(axis=0) for label in unique_labels])
            distance_matrix = cdist(cluster_centers, cluster_centers, metric="cosine")
            np.fill_diagonal(distance_matrix, np.inf)

            min_dist_idx = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
            cluster_to_merge_1, cluster_to_merge_2 = unique_labels[min_dist_idx[0]], unique_labels[min_dist_idx[1]]

            labels[labels == cluster_to_merge_2] = cluster_to_merge_1
            unique_labels = np.unique(labels)
            unique_labels = unique_labels[unique_labels != -1]
            n_clusters = len(unique_labels)

    return n_clusters > 1, labels


def split_tracklets(
    tmp_trklets: Dict[int, Tracklet],
    eps: float,
    max_k: int,
    min_samples: int,
    len_thres: int,
) -> Dict[int, Tracklet]:
    """Split tracklets that contain multiple embedding clusters."""

    new_id = max(tmp_trklets.keys()) + 1 if tmp_trklets else 1
    tracklets: Dict[int, Tracklet] = {}
    for tid in sorted(list(tmp_trklets.keys())):
        trklet = tmp_trklets[tid]
        if len(trklet.times) < len_thres or not trklet.features:
            tracklets[tid] = trklet
            continue

        id_switch_detected, clusters = detect_id_switch(trklet.features, eps=eps, min_samples=min_samples, max_clusters=max_k)
        if not id_switch_detected:
            tracklets[tid] = trklet
            continue

        embs = np.stack(trklet.features)
        frames = np.array(trklet.times)
        bboxes = np.stack(trklet.bboxes)
        scores = np.array(trklet.scores)
        unique_labels = set(clusters)

        for label in unique_labels:
            if label == -1:
                continue
            tmp_embs = embs[clusters == label]
            tmp_frames = frames[clusters == label]
            tmp_bboxes = bboxes[clusters == label]
            tmp_scores = scores[clusters == label]

            tracklets[new_id] = Tracklet(
                new_id,
                tmp_frames.tolist(),
                tmp_scores.tolist(),
                tmp_bboxes.tolist(),
                tmp_embs.tolist(),
                parent_id=tid,
            )
            new_id += 1

    assert len(tracklets) >= len(tmp_trklets)
    return tracklets


def save_results(sct_output_path: Path, tracklets: Dict[int, Tracklet]) -> None:
    """Persist processed tracklets back to a MOT results file."""

    results: List[List[float]] = []

    for i, tid in enumerate(sorted(tracklets.keys())):
        track = tracklets[tid]
        new_tid = i + 1
        for instance_idx, frame_id in enumerate(track.times):
            bbox = track.bboxes[instance_idx]
            score = track.scores[instance_idx] if track.scores else 1.0
            results.append([frame_id, new_tid, bbox[0], bbox[1], bbox[2], bbox[3], score, -1, -1])

    results = sorted(results, key=lambda x: x[0])
    sct_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sct_output_path, "w") as f:
        for line in results:
            f.write(
                f"{int(line[0])},{int(line[1])},{line[2]:.2f},{line[3]:.2f},{line[4]:.2f},{line[5]:.2f},{line[6]:.4f},{int(line[7])},{int(line[8])}\n"
            )
    LOGGER.info(f"Saved SCT results to {sct_output_path}")


def _build_tracklets(tracking_results: np.ndarray) -> Dict[int, Tracklet]:
    """Group detections by track ID and derive lightweight features."""

    tid2track: Dict[int, Tracklet] = {}
    if tracking_results.ndim == 1 and tracking_results.size:
        tracking_results = tracking_results.reshape(1, -1)

    for row in tracking_results:
        frame_id = int(row[0])
        track_id = int(row[1])
        bbox = row[2:6].tolist()
        score = float(row[6]) if row.size > 6 else 1.0
        feature = np.asarray([
            bbox[0] + bbox[2] / 2.0,
            bbox[1] + bbox[3] / 2.0,
            bbox[2],
            bbox[3],
            score,
        ], dtype=np.float32)

        if track_id not in tid2track:
            tid2track[track_id] = Tracklet(track_id, [], [], [], [])
        tid2track[track_id].append_det(frame_id, score, bbox)
        tid2track[track_id].append_feat(feature)

    return tid2track


def process_file(
    file_path: Path,
    use_split: bool,
    use_connect: bool,
    eps: float,
    min_samples: int,
    max_k: int,
    min_len: int,
    spatial_factor: float,
    merge_dist_thres: float,
) -> None:
    """Process a single MOT results file with SCT."""

    if not use_split and not use_connect:
        raise ValueError("At least one of use_split or use_connect must be True.")

    LOGGER.info(f"Applying SCT to: {file_path}")
    tracking_results = np.loadtxt(file_path, delimiter=",")
    if tracking_results.size == 0:
        LOGGER.warning(f"No tracking results in {file_path}. Skipping...")
        return

    tracklets = _build_tracklets(tracking_results)

    if use_split:
        LOGGER.info(f"Splitting {len(tracklets)} tracklets in {file_path.name}")
        tracklets = split_tracklets(tracklets, eps=eps, max_k=max_k, min_samples=min_samples, len_thres=min_len)

    if use_connect:
        max_x_range, max_y_range = get_spatial_constraints(tracklets, spatial_factor)
        dist = get_distance_matrix(tracklets)
        LOGGER.info(f"Merging {len(tracklets)} tracklets in {file_path.name}")
        tracklets = merge_tracklets(
            tracklets,
            dist,
            seq_name=file_path.stem,
            max_x_range=max_x_range,
            max_y_range=max_y_range,
            merge_dist_thres=merge_dist_thres,
        )

    save_results(file_path, tracklets)


def sct(
    mot_results_folder: Path,
    use_split: bool = True,
    use_connect: bool = True,
    eps: float = 0.7,
    min_samples: int = 10,
    max_k: int = 3,
    min_len: int = 40,
    spatial_factor: float = 1.0,
    merge_dist_thres: float = 0.4,
) -> None:
    """Run SCT on all MOT result files in a folder."""

    tracking_files = list(mot_results_folder.glob("MOT*.txt"))
    total_files = len(tracking_files)
    LOGGER.info(f"Found {total_files} file(s) to process with SCT.")
    for file_path in tracking_files:
        process_file(
            file_path,
            use_split=use_split,
            use_connect=use_connect,
            eps=eps,
            min_samples=min_samples,
            max_k=max_k,
            min_len=min_len,
            spatial_factor=spatial_factor,
            merge_dist_thres=merge_dist_thres,
        )


def main() -> None:
    """Command-line entrypoint mirroring :mod:`gsi`."""

    parser = argparse.ArgumentParser(description="Apply Split-and-Connect Tracklets (SCT) to tracking results.")
    parser.add_argument("--path", type=str, required=True, help="Path to MOT results folder")
    args = parser.parse_args()

    mot_results_folder = Path(args.path)
    sct(mot_results_folder)


if __name__ == "__main__":
    main()
