"""Global Tracklet Association post-processing utilities."""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from boxmot.appearance.reid.auto_backend import ReidAutoBackend
from boxmot.utils import WEIGHTS, logger as LOGGER


@dataclass
class Tracklet:
    """Lightweight container for detections, scores, and appearance embeddings."""

    track_id: int
    times: List[int]
    scores: List[float]
    bboxes: List[List[float]]
    features: List[np.ndarray]
    parent_id: int | None = None

    def __post_init__(self) -> None:
        if self.parent_id is None:
            self.parent_id = self.track_id

    def append_det(self, frame_id: int, score: float, bbox: Iterable[float]) -> None:
        """Append a detection to the tracklet."""

        self.times.append(int(frame_id))
        self.scores.append(float(score))
        self.bboxes.append(list(bbox))

    def append_feat(self, feature: np.ndarray) -> None:
        """Append an appearance embedding to the tracklet."""

        self.features.append(np.asarray(feature))

    def extract(self, start_idx: int, end_idx: int) -> "Tracklet":
        """Return a sliced copy of the tracklet."""

        slice_obj = slice(start_idx, end_idx + 1)
        return Tracklet(
            track_id=self.track_id,
            times=self.times[slice_obj],
            scores=self.scores[slice_obj],
            bboxes=self.bboxes[slice_obj],
            features=self.features[slice_obj],
            parent_id=self.parent_id,
        )


# ------------------------
# Stage 1: Tracklet generation
# ------------------------


def _load_tracking_results(prediction_path: Path) -> np.ndarray:
    """Load MOT-format tracking results from a text file."""

    results = np.genfromtxt(prediction_path, dtype=float, delimiter=",")
    if results.size == 0:
        return np.empty((0, 10), dtype=float)
    if results.ndim == 1:
        results = results.reshape(1, -1)
    return results


def _frame_lookup(img_folder: Path) -> Dict[int, Path]:
    """Create a mapping from frame index to image path."""

    frames: Dict[int, Path] = {}
    for img_path in img_folder.iterdir():
        if not img_path.is_file():
            continue
        try:
            frame_idx = int(img_path.stem)
        except ValueError:
            continue
        frames[frame_idx] = img_path
    return frames


def _append_detections(
    seq_tracks: Dict[int, Tracklet],
    frame_id: int,
    frame_results: np.ndarray,
) -> Tuple[List[int], List[List[float]]]:
    """Append detection metadata to tracklets and return xyxy boxes."""

    order: List[int] = []
    boxes: List[List[float]] = []
    for _, track_id, l, t, w, h, score, *_ in frame_results:
        bbox = [l, t, w, h]  # xywh
        score = float(score) if not np.isnan(score) else 1.0
        track_id_int = int(track_id)

        if track_id_int not in seq_tracks:
            seq_tracks[track_id_int] = Tracklet(track_id_int, [], [], [], [])

        seq_tracks[track_id_int].append_det(frame_id, score, bbox)

        order.append(track_id_int)
        boxes.append([l, t, l + w, t + h])

    return order, boxes


def generate_tracklets(
    model_path: Path,
    data_path: Path,
    pred_dir: Path,
    tracker: str,
    device: str = "",
    half: bool = False,
) -> Path:
    """Generate pickled tracklets enriched with appearance embeddings.

    The ``pred_dir`` argument should point to the MOT experiment directory
    (for example ``runs/mot/exp``) that already contains the per-sequence
    MOT-format ``.txt`` prediction files. Tracklets will be written inside the
    same experiment folder.
    """

    data_path = data_path.expanduser().resolve()
    pred_dir = pred_dir.expanduser().resolve()
    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")

    split = data_path.name

    reid_backend = ReidAutoBackend(weights=model_path, device=device, half=half).model
    reid_backend.warmup()

    output_dir = pred_dir / f"{tracker}_Tracklets_{split}"
    output_dir.mkdir(parents=True, exist_ok=True)

    seqs = sorted(pred_dir.glob("*.txt"))
    for seq_file in tqdm(seqs, desc="Processing sequences"):
        seq_name = seq_file.stem

        img_dir = data_path / seq_name / "img1"
        if not img_dir.exists():
            LOGGER.warning(f"Image directory missing for sequence {seq_name}: {img_dir}")
            continue

        frame_paths = _frame_lookup(img_dir)

        tracking_results = _load_tracking_results(seq_file)
        if tracking_results.size == 0:
            LOGGER.warning(f"No tracking results found in {seq_file}")
            continue

        last_frame = int(tracking_results[:, 0].max())
        seq_tracks: Dict[int, Tracklet] = {}

        for frame_id in range(1, last_frame + 1):
            frame_mask = tracking_results[:, 0] == frame_id
            frame_res = tracking_results[frame_mask]
            if frame_res.size == 0:
                continue

            img_path = frame_paths.get(frame_id)
            if img_path is None:
                LOGGER.warning(f"Missing frame {frame_id} for sequence {seq_name}")
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                LOGGER.warning(f"Unable to read image: {img_path}")
                continue

            track_order, xyxys = _append_detections(seq_tracks, frame_id, frame_res)
            xyxys_array = np.asarray(xyxys, dtype=np.float32)

            features = reid_backend.get_features(xyxys_array, img)
            for idx, track_id in enumerate(track_order):
                seq_tracks[track_id].append_feat(features[idx])

        track_output_path = output_dir / f"{seq_name}.pkl"
        with track_output_path.open("wb") as f:
            pickle.dump(seq_tracks, f)
        LOGGER.info(f"Saved tracklets to {track_output_path}")

    return output_dir


# ------------------------
# Stage 2: Refinement
# ------------------------


def find_consecutive_segments(track_times: List[int]) -> List[Tuple[int, int]]:
    """Return (start_index, end_index) for each consecutive segment in frame ids."""

    if not track_times:
        return []

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
    seg1: List[Tuple[int, int]],
    seg2: List[Tuple[int, int]],
    track1: Tracklet,
    track2: Tracklet,
) -> List[Tracklet]:
    """Pair segments from two tracks to form temporally sorted subtracks."""

    subtracks: List[Tracklet] = []

    seg1 = list(seg1)
    seg2 = list(seg2)

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
    """Compute maximal motion ranges (x_range, y_range) across tracklets."""

    min_x = float("inf")
    max_x = float("-inf")
    min_y = float("inf")
    max_y = float("-inf")

    for track in tid2track.values():
        for bbox in track.bboxes:
            x, y, w, h = bbox[0:4]
            x += w / 2.0
            y += h / 2.0
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

    x_range = abs(max_x - min_x) * factor
    y_range = abs(max_y - min_y) * factor

    return x_range, y_range


def get_distance(
    track1_id: int,
    track2_id: int,
    track1: Tracklet,
    track2: Tracklet,
) -> float:
    """Cosine distance between two tracklets, penalizing temporal overlap."""

    assert track1_id == track1.track_id and track2_id == track2.track_id

    if track1_id != track2_id:
        if set(track1.times) & set(track2.times):
            return 1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    track1_features_tensor = torch.tensor(np.stack(track1.features), dtype=torch.float32, device=device)
    track2_features_tensor = torch.tensor(np.stack(track2.features), dtype=torch.float32, device=device)

    count1 = len(track1_features_tensor)
    count2 = len(track2_features_tensor)

    cos_sim_numerator = torch.matmul(track1_features_tensor, track2_features_tensor.T)
    track1_norms = torch.norm(track1_features_tensor, p=2, dim=1, keepdim=True)
    track2_norms = torch.norm(track2_features_tensor, p=2, dim=1, keepdim=True)
    cos_sim_denominator = torch.matmul(track1_norms, track2_norms.T)

    cos_dist = 1.0 - cos_sim_numerator / cos_sim_denominator.clamp(min=1e-8)
    total_cos_dist = cos_dist.sum()
    result = total_cos_dist / (count1 * count2)

    return float(result.item())


def get_distance_matrix(tid2track: Dict[int, Tracklet]) -> np.ndarray:
    """Build a symmetric distance matrix between all tracklets."""

    n = len(tid2track)
    Dist = np.zeros((n, n), dtype=np.float32)

    items = list(tid2track.items())
    for i, (track1_id, track1) in enumerate(items):
        assert len(track1.times) == len(track1.bboxes)
        for j, (track2_id, track2) in enumerate(items):
            if j < i:
                Dist[i, j] = Dist[j, i]
                continue
            Dist[i, j] = get_distance(track1_id, track2_id, track1, track2)

    return Dist


def check_spatial_constraints(
    trk_1: Tracklet,
    trk_2: Tracklet,
    max_x_range: float,
    max_y_range: float,
) -> bool:
    """Check if two tracklets are spatially consistent for merging."""

    in_spatial_range = True
    seg_1 = find_consecutive_segments(trk_1.times)
    seg_2 = find_consecutive_segments(trk_2.times)

    subtracks = query_subtracks(seg_1, seg_2, trk_1, trk_2)
    if not subtracks:
        return False

    subtrack_1st = subtracks.pop(0)

    while subtracks:
        subtrack_2nd = subtracks.pop(0)

        if subtrack_1st.parent_id == subtrack_2nd.parent_id:
            subtrack_1st = subtrack_2nd
            continue

        x_1, y_1, w_1, h_1 = subtrack_1st.bboxes[-1][0:4]
        x_2, y_2, w_2, h_2 = subtrack_2nd.bboxes[0][0:4]
        x_1 += w_1 / 2.0
        y_1 += h_1 / 2.0
        x_2 += w_2 / 2.0
        y_2 += h_2 / 2.0

        dx = abs(x_1 - x_2)
        dy = abs(y_1 - y_2)

        if dx > max_x_range or dy > max_y_range:
            in_spatial_range = False
            break
        subtrack_1st = subtrack_2nd

    return in_spatial_range


def merge_tracklets(
    tracklets: Dict[int, Tracklet],
    Dist: np.ndarray,
    seq_name: str | None,
    max_x_range: float,
    max_y_range: float,
    merge_dist_thres: float,
) -> Dict[int, Tracklet]:
    """Merge spatially consistent tracklets whose appearance distance is small."""

    idx2tid = {idx: tid for idx, tid in enumerate(tracklets.keys())}

    diagonal_mask = np.eye(Dist.shape[0], dtype=bool)
    non_diagonal_mask = ~diagonal_mask

    while np.any(Dist[non_diagonal_mask] < merge_dist_thres):
        min_index = np.argmin(Dist[non_diagonal_mask])
        masked_indices = np.where(non_diagonal_mask)
        track1_idx, track2_idx = masked_indices[0][min_index], masked_indices[1][min_index]

        track1 = tracklets[idx2tid[track1_idx]]
        track2 = tracklets[idx2tid[track2_idx]]

        in_spatial_range = check_spatial_constraints(track1, track2, max_x_range, max_y_range)

        if in_spatial_range:
            track1.features += track2.features
            track1.times += track2.times
            track1.bboxes += track2.bboxes

            tracklets[idx2tid[track1_idx]] = track1
            tracklets.pop(idx2tid[track2_idx])

            Dist = np.delete(Dist, track2_idx, axis=0)
            Dist = np.delete(Dist, track2_idx, axis=1)

            idx2tid = {idx: tid for idx, tid in enumerate(tracklets.keys())}

            for idx in range(Dist.shape[0]):
                d = get_distance(
                    idx2tid[track1_idx],
                    idx2tid[idx],
                    tracklets[idx2tid[track1_idx]],
                    tracklets[idx2tid[idx]],
                )
                Dist[track1_idx, idx] = d
                Dist[idx, track1_idx] = d

            diagonal_mask = np.eye(Dist.shape[0], dtype=bool)
            non_diagonal_mask = ~diagonal_mask
        else:
            Dist[track1_idx, track2_idx] = merge_dist_thres
            Dist[track2_idx, track1_idx] = merge_dist_thres

    return tracklets


def detect_id_switch(
    embs: List[np.ndarray],
    eps: float,
    min_samples: int,
    max_clusters: int | None,
) -> Tuple[bool, np.ndarray]:
    """Cluster embeddings with DBSCAN and decide if an ID switch is present."""

    if len(embs) > 15000:
        embs = embs[1::2]

    embs = np.stack(embs)

    scaler = StandardScaler()
    embs_scaled = scaler.fit_transform(embs)

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
            c1, c2 = unique_labels[min_dist_idx[0]], unique_labels[min_dist_idx[1]]

            labels[labels == c2] = c1
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
    """Split tracklets using DBSCAN-based ID-switch detection."""

    if not tmp_trklets:
        return {}

    new_id = max(tmp_trklets.keys()) + 1
    tracklets: Dict[int, Tracklet] = {}

    for tid in tqdm(sorted(tmp_trklets.keys()), total=len(tmp_trklets), desc="Splitting tracklets"):
        trklet = tmp_trklets[tid]

        if len(trklet.times) < len_thres:
            tracklets[tid] = trklet
            continue

        embs = np.stack(trklet.features)
        frames = np.array(trklet.times)
        bboxes = np.stack(trklet.bboxes)
        scores = np.array(trklet.scores)

        id_switch_detected, clusters = detect_id_switch(
            list(embs),
            eps=eps,
            min_samples=min_samples,
            max_clusters=max_k,
        )

        if not id_switch_detected:
            tracklets[tid] = trklet
            continue

        for label in set(clusters):
            if label == -1:
                continue

            mask = clusters == label
            tmp_embs = embs[mask]
            tmp_frames = frames[mask]
            tmp_bboxes = bboxes[mask]
            tmp_scores = scores[mask]

            new_trk = Tracklet(
                new_id,
                tmp_frames.tolist(),
                tmp_scores.tolist(),
                tmp_bboxes.tolist(),
                tmp_embs.tolist(),
                parent_id=trklet.track_id,
            )

            tracklets[new_id] = new_trk
            new_id += 1

    return tracklets


def save_results(sct_output_path: Path, tracklets: Dict[int, Tracklet]) -> None:
    """Save final merged tracklets back to MOT txt format."""

    results: List[List[float]] = []

    for i, tid in enumerate(sorted(tracklets.keys())):
        track = tracklets[tid]
        new_tid = i + 1
        for instance_idx, frame_id in enumerate(track.times):
            bbox = track.bboxes[instance_idx]
            results.append(
                [frame_id, new_tid, bbox[0], bbox[1], bbox[2], bbox[3], 1, -1, -1, -1]
            )

    results = sorted(results, key=lambda x: x[0])

    lines = []
    for line in results:
        lines.append(
            f"{line[0]},{line[1]},{line[2]:.2f},{line[3]:.2f},{line[4]:.2f},{line[5]:.2f},{line[6]},{line[7]},{line[8]},{line[9]}\n"
        )

    sct_output_path.parent.mkdir(parents=True, exist_ok=True)
    with sct_output_path.open("w") as f:
        f.writelines(lines)

    LOGGER.info(f"Saved SCT results to {sct_output_path}")


def refine_tracklets(
    track_src: Path,
    tracker: str,
    dataset: str,
    use_split: bool = True,
    min_len: int = 100,
    eps: float = 0.6,
    min_samples: int = 10,
    max_k: int = 3,
    use_connect: bool = True,
    spatial_factor: float = 1.0,
    merge_dist_thres: float = 0.4,
) -> Path:
    """Run the splitting and connecting stages over saved tracklets."""

    seqs_tracks = sorted(track_src.iterdir())
    data_path = track_src.parent

    for seq_idx, seq in enumerate(seqs_tracks):
        seq_name = seq.stem
        LOGGER.info(f"Processing seq {seq_idx + 1} / {len(seqs_tracks)}: {seq_name}")

        with seq.open("rb") as pkl_f:
            tmp_trklets: Dict[int, Tracklet] = pickle.load(pkl_f)

        max_x_range, max_y_range = get_spatial_constraints(tmp_trklets, spatial_factor)

        if use_split:
            LOGGER.info(f"Number of tracklets before splitting: {len(tmp_trklets)}")
            split_tracklets_dict = split_tracklets(
                tmp_trklets,
                eps=eps,
                max_k=max_k,
                min_samples=min_samples,
                len_thres=min_len,
            )
        else:
            split_tracklets_dict = tmp_trklets

        Dist = get_distance_matrix(split_tracklets_dict)
        LOGGER.info(f"Number of tracklets before merging: {len(split_tracklets_dict)}")

        if use_connect:
            merged_tracklets = merge_tracklets(
                split_tracklets_dict,
                Dist,
                seq_name=seq_name,
                max_x_range=max_x_range,
                max_y_range=max_y_range,
                merge_dist_thres=merge_dist_thres,
            )
        else:
            merged_tracklets = split_tracklets_dict

        LOGGER.info(f"Number of tracklets after merging: {len(merged_tracklets)}")

        process = []
        if use_split:
            process.append("Split")
        if use_connect:
            process.append("Connect")
        process_str = "+".join(process)

        sct_name = (
            f"{tracker}_{dataset}_{process_str}"
            f"_eps{eps}_minSamples{min_samples}"
            f"_K{max_k}_mergeDist{merge_dist_thres}"
            f"_spatial{spatial_factor}"
        )

        new_sct_output_dir = data_path / sct_name
        new_sct_output_path = new_sct_output_dir / f"{seq_name}.txt"
        save_results(new_sct_output_path, merged_tracklets)

    return data_path / sct_name


# ------------------------
# Public API / CLI
# ------------------------


def gta(
    mot_results_folder: Path,
    data_path: Path,
    model_path: Path = WEIGHTS / "lmbn_n_duke.pt",
    tracker: str = "deepocsort",
    device: str = "",
    half: bool = False,
    use_split: bool = True,
    min_len: int = 100,
    eps: float = 0.6,
    min_samples: int = 10,
    max_k: int = 3,
    use_connect: bool = True,
    spatial_factor: float = 1.0,
    merge_dist_thres: float = 0.4,
) -> None:
    """Run the full Global Tracklet Association pipeline on MOT results."""

    tracklet_dir = generate_tracklets(
        model_path=model_path,
        data_path=data_path,
        pred_dir=mot_results_folder,
        tracker=tracker,
        device=device,
        half=half,
    )

    refine_tracklets(
        track_src=tracklet_dir,
        tracker=tracker,
        dataset=data_path.name,
        use_split=use_split,
        min_len=min_len,
        eps=eps,
        min_samples=min_samples,
        max_k=max_k,
        use_connect=use_connect,
        spatial_factor=spatial_factor,
        merge_dist_thres=merge_dist_thres,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Global Tracklet Association pipeline")
    subparsers = parser.add_subparsers(dest="stage", required=True)

    gen_parser = subparsers.add_parser("generate", help="Generate tracklets with embeddings")
    gen_parser.add_argument("--model_path", type=Path, default=Path("lmbn_n_duke.pt"))
    gen_parser.add_argument("--data_path", type=Path, default=Path("boxmot/engine/data/MOT17-ablation/train"))
    gen_parser.add_argument(
        "--pred_dir",
        type=Path,
        default=Path("runs/mot/exp"),
        help="Path to the MOT experiment directory containing prediction .txt files.",
    )
    gen_parser.add_argument("--tracker", type=str, default="deepocsort")
    gen_parser.add_argument("--device", type=str, default="mps")
    gen_parser.add_argument("--half", action="store_true")

    ref_parser = subparsers.add_parser("refine", help="Refine and merge tracklets")
    ref_parser.add_argument("--dataset", type=str, default="SportsMOT")
    ref_parser.add_argument("--tracker", type=str, default="deepocsort")
    ref_parser.add_argument(
        "--track_src",
        type=Path,
        default=Path("runs/mot/exp/deepocsort_Tracklets_train"),
        help="Directory of generated tracklet pickles (inside the experiment folder).",
    )
    ref_parser.add_argument("--use_split", action="store_true")
    ref_parser.add_argument("--min_len", type=int, default=100)
    ref_parser.add_argument("--eps", type=float, default=0.6)
    ref_parser.add_argument("--min_samples", type=int, default=10)
    ref_parser.add_argument("--max_k", type=int, default=3)
    ref_parser.add_argument("--use_connect", action="store_true")
    ref_parser.add_argument("--spatial_factor", type=float, default=1.0)
    ref_parser.add_argument("--merge_dist_thres", type=float, default=0.4)

    args = parser.parse_args()

    if args.stage == "generate":
        generate_tracklets(
            model_path=args.model_path,
            data_path=args.data_path,
            pred_dir=args.pred_dir,
            tracker=args.tracker,
            device=args.device,
            half=args.half,
        )
    else:
        refine_tracklets(
            track_src=args.track_src,
            tracker=args.tracker,
            dataset=args.dataset,
            use_split=args.use_split,
            min_len=args.min_len,
            eps=args.eps,
            min_samples=args.min_samples,
            max_k=args.max_k,
            use_connect=args.use_connect,
            spatial_factor=args.spatial_factor,
            merge_dist_thres=args.merge_dist_thres,
        )
