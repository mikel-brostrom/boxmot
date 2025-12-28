# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

import argparse
import concurrent.futures
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from boxmot.utils import logger as LOGGER


class Tracklet:
    def __init__(self, track_id=None, frames=None, scores=None, bboxes=None, feats=None):
        '''
        Initialize the Tracklet with IDs, times, scores, bounding boxes, and optional features.
        If parameters are not provided, initializes them to None or empty lists.

        Args:
            track_id (int, optional): Unique identifier for the track. Defaults to None.
            frames (list or int, optional): Frame numbers where the track is present. Can be a list of frames or a single frame. Defaults to None.
            scores (list or float, optional): Detection scores corresponding to frames. Can be a list of scores or a single score. Defaults to None.
            bboxes (list of lists or list, optional): Bounding boxes corresponding to each frame. Each bounding box is a list of 4 elements. Defaults to None.
            feats (list of np.array, optional): Feature vectors corresponding to frames. Each feature should be a numpy array of shape (512,). Defaults to None.
        '''
        self.track_id = track_id
        self.parent_id = track_id
        self.scores = scores if isinstance(scores, list) else [scores] if scores is not None else []
        self.times = frames if isinstance(frames, list) else [frames] if frames is not None else []
        self.bboxes = bboxes if isinstance(bboxes, list) and bboxes and isinstance(bboxes[0], list) else [bboxes] if bboxes is not None else []
        self.features = feats if feats is not None else []

    def append_det(self, frame, score, bbox):
        '''
        Appends a detection to the tracklet.

        Args:
            frame (int): Frame number for the detection.
            score (float): Detection score.
            bbox (list of float): Bounding box with four elements [x, y, width, height].
        '''
        self.scores.append(score)
        self.times.append(frame)
        self.bboxes.append(bbox)

    def append_feat(self, feat):
        '''
        Appends a feature vector to the tracklet.

        Args:
            feat (np.array): Feature vector of shape (512,).
        '''
        self.features.append(feat)

    def extract(self, start, end):
        '''
        Extracts a subtrack from the tracklet between two indices.

        Args:
            start (int): Start index for the extraction.
            end (int): End index for the extraction.

        Returns:
            Tracklet: A new Tracklet object that is a subset of the original from start to end indices.
        '''
        subtrack = Tracklet(self.track_id, self.times[start:end + 1], self.scores[start:end + 1], self.bboxes[start:end + 1], self.features[start:end + 1] if self.features else None)
        return subtrack


def find_consecutive_segments(track_times):
    """
    Identifies and returns the start and end indices of consecutive segments in a list of times.

    Args:
        track_times (list): A list of frame times (integers) representing when a tracklet was detected.

    Returns:
        list of tuples: Each tuple contains two integers (start_index, end_index) representing the start and end of a consecutive segment.
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


def query_subtracks(seg1, seg2, track1, track2):
    """
    Processes and pairs up segments from two different tracks to form valid subtracks based on their temporal alignment.

    Args:
        seg1 (list of tuples): List of segments from the first track where each segment is a tuple of start and end indices.
        seg2 (list of tuples): List of segments from the second track similar to seg1.
        track1 (Tracklet): First track object containing times and bounding boxes.
        track2 (Tracklet): Second track object similar to track1.

    Returns:
        list: Returns a list of subtracks which are either segments of track1 or track2 sorted by time.
    """
    subtracks = []  # List to store valid subtracks
    while seg1 and seg2:  # Continue until seg1 or seg1 is empty
        s1_start, s1_end = seg1[0]  # Get the start and end indices of the first segment in seg1
        s2_start, s2_end = seg2[0]  # Get the start and end indices of the first segment in seg2
        
        subtrack_1 = track1.extract(s1_start, s1_end)
        subtrack_2 = track2.extract(s2_start, s2_end)

        s1_startFrame = track1.times[s1_start]  # Get the starting frame of subtrack 1
        s2_startFrame = track2.times[s2_start]  # Get the starting frame of subtrack 2

        if s1_startFrame < s2_startFrame:  # Compare the starting frames of the two subtracks
            # assert track1.times[s1_end] <= s2_startFrame
            subtracks.append(subtrack_1)
            subtracks.append(subtrack_2)
        else:
            # assert s1_startFrame >= track2.times[s2_end]
            subtracks.append(subtrack_2)
            subtracks.append(subtrack_1)
        seg1.pop(0)
        seg2.pop(0)
    
    seg_remain = seg1 if seg1 else seg2
    track_remain = track1 if seg1 else track2
    while seg_remain:
        s_start, s_end = seg_remain[0]
        if(s_end - s_start) < 30:
            seg_remain.pop(0)
            continue
        subtracks.append(track_remain.extract(s_start, s_end))
        seg_remain.pop(0)
    
    return subtracks  # Return the list of valid subtracks sorted ascending temporally


def get_spatial_constraints(tid2track, factor):
    """
    Calculates and returns the maximal spatial constraints for bounding boxes across all tracks.

    Args:
        tid2track (dict): Dictionary mapping track IDs to their respective track objects.
        factor (float): Factor by which to scale the calculated x and y ranges.

    Returns:
        tuple: Maximal x and y range scaled by the given factor.
    """

    min_x = float('inf')
    max_x = -float('inf')
    min_y = float('inf')
    max_y = -float('inf')

    for track in tid2track.values():
        for bbox in track.bboxes:
            assert len(bbox) == 4
            x, y, w, h = bbox[0:4]  # x, y is coordinate of top-left point of bounding box
            x += w / 2  # get center point
            y += h / 2  # get center point
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

    x_range = abs(max_x - min_x) * factor
    y_range = abs(max_y - min_y) * factor

    return x_range, y_range


def get_distance_matrix(tid2track):
    """
    Constructs and returns a distance matrix between all tracklets based on overlapping times and feature similarities.

    Args:
        tid2track (dict): Dictionary mapping track IDs to their respective track objects.

    Returns:
        ndarray: A square matrix where each element (i, j) represents the calculated distance between track i and track j.
    """
    Dist = np.zeros((len(tid2track), len(tid2track)))

    for i, (track1_id, track1) in enumerate(tid2track.items()):
        assert len(track1.times) == len(track1.bboxes)
        for j, (track2_id, track2) in enumerate(tid2track.items()):
            if j < i:
                Dist[i][j] = Dist[j][i]
            else:
                Dist[i][j] = get_distance(track1_id, track2_id, track1, track2)
    return Dist


def get_distance(track1_id, track2_id, track1, track2):
    """
    Calculates the cosine distance between two tracks using PyTorch for efficient computation.

    Args:
        track1_id (int): ID of the first track.
        track2_id (int): ID of the second track.
        track1 (Tracklet): First track object.
        track2 (Tracklet): Second track object.

    Returns:
        float: Cosine distance between the two tracks.
    """
    assert track1_id == track1.track_id and track2_id == track2.track_id   # debug line
    doesOverlap = False
    if (track1_id != track2_id):
        doesOverlap = set(track1.times) & set(track2.times)
    if doesOverlap:
        return 1                # make the cosine distance between two tracks maximum, max = 1
    else:
        # calculate cosine distance between two tracks based on features
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        track1_features_tensor = torch.tensor(np.stack(track1.features), dtype=torch.float32).to(device)
        track2_features_tensor = torch.tensor(np.stack(track2.features), dtype=torch.float32).to(device)
        count1 = len(track1_features_tensor)
        count2 = len(track2_features_tensor)

        cos_sim_Numerator = torch.matmul(track1_features_tensor, track2_features_tensor.T)
        track1_features_dist = torch.norm(track1_features_tensor, p=2, dim=1, keepdim=True)
        track2_features_dist = torch.norm(track2_features_tensor, p=2, dim=1, keepdim=True)
        cos_sim_Denominator = torch.matmul(track1_features_dist, track2_features_dist.T)
        cos_Dist = 1 - cos_sim_Numerator / cos_sim_Denominator
        
        total_cos_Dist = cos_Dist.sum()
        result = total_cos_Dist / (count1 * count2)
        return result.item()


def check_spatial_constraints(trk_1, trk_2, max_x_range, max_y_range):
    """
    Checks if two tracklets meet spatial constraints for potential merging.

    Args:
        trk_1 (Tracklet): The first tracklet object containing times and bounding boxes.
        trk_2 (Tracklet): The second tracklet object containing times and bounding boxes, to be evaluated
                        against trk_1 for merging possibility.
        max_x_range (float): The maximum allowed distance in the x-coordinate between the end of trk_1 and
                             the start of trk_2 for them to be considered for merging.
        max_y_range (float): The maximum allowed distance in the y-coordinate under the same conditions as
                             the x-coordinate.

    Returns:
        bool: True if the spatial constraints are met (the tracklets are close enough to consider merging),
              False otherwise.
    """
    inSpatialRange = True
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
        x_1, y_1, w_1, h_1 = subtrack_1st.bboxes[-1][0 : 4]
        x_2, y_2, w_2, h_2 = subtrack_2nd.bboxes[0][0 : 4]
        x_1 += w_1 / 2
        y_1 += h_1 / 2
        x_2 += w_2 / 2
        y_2 += h_2 / 2
        dx = abs(x_1 - x_2)
        dy = abs(y_1 - y_2)
        
        # check the distance between exit location of track_1 and enter location of track_2
        if dx > max_x_range or dy > max_y_range:
            inSpatialRange = False
            break
        else:
            subtrack_1st = subtrack_2nd
    return inSpatialRange


def merge_tracklets(tracklets, Dist, max_x_range=None, max_y_range=None, merge_dist_thres=None):
    idx2tid = {idx: tid for idx, tid in enumerate(tracklets.keys())}
    
    diagonal_mask = np.eye(Dist.shape[0], dtype=bool)
    non_diagonal_mask = ~diagonal_mask
    
    while (np.any(Dist[non_diagonal_mask] < merge_dist_thres)):
        min_index = np.argmin(Dist[non_diagonal_mask])
        min_value = np.min(Dist[non_diagonal_mask])
        masked_indices = np.where(non_diagonal_mask)
        track1_idx, track2_idx = masked_indices[0][min_index], masked_indices[1][min_index]

        track1 = tracklets[idx2tid[track1_idx]]
        track2 = tracklets[idx2tid[track2_idx]]

        inSpatialRange = check_spatial_constraints(track1, track2, max_x_range, max_y_range)
        
        if inSpatialRange:
            track1.features += track2.features
            track1.times += track2.times
            track1.bboxes += track2.bboxes
            track1.scores += track2.scores
            
            # update tracklets dictionary
            tracklets[idx2tid[track1_idx]] = track1
            tracklets.pop(idx2tid[track2_idx])

            # Remove the merged tracklet (track2) from the distance matrix
            Dist = np.delete(Dist, track2_idx, axis=0)  # Remove row for track2
            Dist = np.delete(Dist, track2_idx, axis=1)  # Remove column for track2
            # update idx2tid
            idx2tid = {idx: tid for idx, tid in enumerate(tracklets.keys())}
            
            # Update distance matrix only for the merged tracklet's row and column
            for idx in range(Dist.shape[0]):
                Dist[track1_idx, idx] = get_distance(idx2tid[track1_idx], idx2tid[idx], tracklets[idx2tid[track1_idx]], tracklets[idx2tid[idx]])
                Dist[idx, track1_idx] = Dist[track1_idx, idx]  # Ensure symmetry
            
            # update mask
            diagonal_mask = np.eye(Dist.shape[0], dtype=bool)
            non_diagonal_mask = ~diagonal_mask
        else:
            # change distance between track pair to threshold
            Dist[track1_idx, track2_idx], Dist[track2_idx, track1_idx] = merge_dist_thres, merge_dist_thres
    
    return tracklets


def detect_id_switch(embs, eps=None, min_samples=None, max_clusters=None):
    """
    Detects identity switches within a tracklet using clustering.
    """
    if len(embs) > 15000:
        embs = embs[1::2]

    embs = np.stack(embs)
    
    # Standardize the embeddings
    scaler = StandardScaler()
    embs_scaled = scaler.fit_transform(embs)

    # Apply DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(embs_scaled)
    labels = db.labels_

    # Count the number of clusters (excluding noise)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]

    if -1 in labels and len(unique_labels) > 1:
        # Find the cluster centers
        cluster_centers = np.array([embs_scaled[labels == label].mean(axis=0) for label in unique_labels])
        
        # Assign noise points to the nearest cluster
        noise_indices = np.where(labels == -1)[0]
        for idx in noise_indices:
            distances = cdist([embs_scaled[idx]], cluster_centers, metric='cosine')
            nearest_cluster = np.argmin(distances)
            labels[idx] = list(unique_labels)[nearest_cluster]
    
    n_clusters = len(unique_labels)

    if max_clusters and n_clusters > max_clusters:
        # Merge clusters to ensure the number of clusters does not exceed max_clusters
        while n_clusters > max_clusters:
            cluster_centers = np.array([embs_scaled[labels == label].mean(axis=0) for label in unique_labels])
            distance_matrix = cdist(cluster_centers, cluster_centers, metric='cosine')
            np.fill_diagonal(distance_matrix, np.inf)  # Ignore self-distances
            
            # Find the closest pair of clusters
            min_dist_idx = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
            cluster_to_merge_1, cluster_to_merge_2 = unique_labels[min_dist_idx[0]], unique_labels[min_dist_idx[1]]

            # Merge the clusters
            labels[labels == cluster_to_merge_2] = cluster_to_merge_1
            unique_labels = np.unique(labels)
            unique_labels = unique_labels[unique_labels != -1]
            n_clusters = len(unique_labels)

    return n_clusters > 1, labels


def split_tracklets(tmp_trklets, eps=None, max_k=None, min_samples=None, len_thres=None):
    """
    Splits each tracklet into multiple tracklets based on an internal distance threshold.
    """
    new_id = max(tmp_trklets.keys()) + 1
    tracklets = defaultdict()
    # Splitting algorithm to process every tracklet in a sequence
    for tid in sorted(list(tmp_trklets.keys())):
        trklet = tmp_trklets[tid]
        if len(trklet.times) < len_thres:  # NOTE: Set tracklet length threshold to filter out short ones
            tracklets[tid] = trklet
        else:
            embs = np.stack(trklet.features)
            frames = np.array(trklet.times)
            bboxes = np.stack(trklet.bboxes)
            scores = np.array(trklet.scores)
            # Perform DBSCAN clustering
            id_switch_detected, clusters = detect_id_switch(embs, eps=eps, min_samples=min_samples, max_clusters=max_k)
            if not id_switch_detected:
                tracklets[tid] = trklet
            else:
                unique_labels = set(clusters)

                for label in unique_labels:
                    if label == -1:
                        continue  # Skip noise points
                    tmp_embs = embs[clusters == label]
                    tmp_frames = frames[clusters == label]
                    tmp_bboxes = bboxes[clusters == label]
                    tmp_scores = scores[clusters == label]
                    assert new_id not in tmp_trklets
                    
                    tracklets[new_id] = Tracklet(new_id, tmp_frames.tolist(), tmp_scores.tolist(), tmp_bboxes.tolist(), feats=tmp_embs.tolist())
                    new_id += 1

    assert len(tracklets) >= len(tmp_trklets)
    return tracklets


def iou_batch(bboxes1, bboxes2):
    """
    Compute IoU between two batches of bounding boxes.
    bboxes1: [N, 4] (x1, y1, x2, y2)
    bboxes2: [M, 4] (x1, y1, x2, y2)
    Returns: [N, M] IoU matrix
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)
    
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    
    o = wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1]) +
              (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)
    return o


def process_sequence(mot_file, dets_file, embs_file, eps, min_samples, max_k, min_len, spatial_factor, merge_dist_thres):
    LOGGER.info(f"Processing {mot_file.name}")
    
    # Load MOT results
    # Format: frame, id, x, y, w, h, conf, -1, -1, -1
    mot_data = np.loadtxt(mot_file, delimiter=',')
    if mot_data.size == 0:
        return
    if mot_data.ndim == 1:
        mot_data = mot_data.reshape(1, -1)
    
    # Sort by frame
    mot_data = mot_data[np.argsort(mot_data[:, 0])]

    # Load Dets and Embs
    # Dets: frame, x1, y1, x2, y2, conf, cls
    dets_data = np.loadtxt(dets_file)
    if dets_data.ndim == 1:
        dets_data = dets_data.reshape(1, -1)

    # Embs: embedding vector
    embs_data = np.loadtxt(embs_file)
    if embs_data.ndim == 1:
        embs_data = embs_data.reshape(1, -1)
    
    if dets_data.size == 0 or embs_data.size == 0:
        LOGGER.warning(f"No dets or embs for {mot_file.name}")
        return

    # Build Tracklets
    tracklets = {}
    
    # Group dets by frame for faster lookup
    dets_by_frame = defaultdict(list)
    for i, det in enumerate(dets_data):
        frame = int(det[0])
        dets_by_frame[frame].append((i, det))

    # Iterate over MOT results and assign embeddings
    num_cols = mot_data.shape[1]
    for row in mot_data:
        frame = int(row[0])
        tid = int(row[1])
        bbox_mot = row[2:6] # x, y, w, h
        
        if num_cols == 9:
            conf = row[8]
        elif num_cols == 10:
            conf = row[6]
        else:
            conf = 1.0
        
        # Convert MOT bbox to x1, y1, x2, y2
        bbox_mot_xyxy = np.array([bbox_mot[0], bbox_mot[1], bbox_mot[0] + bbox_mot[2], bbox_mot[1] + bbox_mot[3]])
        
        # Find matching detection
        best_iou = 0
        best_idx = -1
        
        if frame in dets_by_frame:
            for i, det in dets_by_frame[frame]:
                bbox_det = det[1:5] # x1, y1, x2, y2
                
                # Calculate IoU
                iou = iou_batch(bbox_mot_xyxy.reshape(1, 4), bbox_det.reshape(1, 4))[0][0]
                
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
        
        if best_iou > 0.7 and best_idx != -1:
            emb = embs_data[best_idx]
            
            if tid not in tracklets:
                tracklets[tid] = Tracklet(tid)
            
            tracklets[tid].append_det(frame, conf, bbox_mot.tolist())
            tracklets[tid].append_feat(emb)
    
    # Filter tracklets that have features
    tracklets = {tid: t for tid, t in tracklets.items() if len(t.features) > 0}
    
    if not tracklets:
        LOGGER.warning(f"No tracklets matched for {mot_file.name}")
        return

    # Split
    LOGGER.info(f"Splitting tracklets for {mot_file.name}...")
    split_tracklets_dict = split_tracklets(tracklets, eps=eps, max_k=max_k, min_samples=min_samples, len_thres=min_len)
    
    # Connect
    LOGGER.info(f"Merging tracklets for {mot_file.name}...")
    max_x_range, max_y_range = get_spatial_constraints(split_tracklets_dict, spatial_factor)
    Dist = get_distance_matrix(split_tracklets_dict)
    merged_tracklets = merge_tracklets(split_tracklets_dict, Dist, max_x_range=max_x_range, max_y_range=max_y_range, merge_dist_thres=merge_dist_thres)
    
    # Save results
    results = []
    for i, tid in enumerate(sorted(merged_tracklets.keys())):
        track = merged_tracklets[tid]
        new_tid = i + 1
        for instance_idx, frame_id in enumerate(track.times):
            bbox = track.bboxes[instance_idx]
            results.append(
                [frame_id, new_tid, bbox[0], bbox[1], bbox[2], bbox[3], track.scores[instance_idx], -1, -1, -1]
            )
    
    results = sorted(results, key=lambda x: x[0])
    
    # Overwrite the original MOT file with the SCT results
    with open(mot_file, 'w') as f:
        for row in results:
            # frame, id, x, y, w, h, not_ignored, class, conf
            line = f"{int(row[0])},{int(row[1])},{row[2]:.2f},{row[3]:.2f},{row[4]:.2f},{row[5]:.2f},1,1,{row[6]:.6f}\n"
            f.write(line)
            
    LOGGER.info(f"Saved SCT results to {mot_file}")


def sct(mot_results_folder: Path, dets_folder: Path, embs_folder: Path, 
        eps=0.05, min_samples=10, max_k=20, min_len=100, spatial_factor=0.5, merge_dist_thres=0.3):
    """
    Apply Split and Connect Tracking (SCT) post-processing to MOT results.

    Args:
        mot_results_folder (Path): Path to the folder containing MOT result files.
        dets_folder (Path): Path to the folder containing detection files.
        embs_folder (Path): Path to the folder containing embedding files.
        eps (float, optional): The maximum distance between two samples for one to be considered as in the neighborhood of the other. Defaults to 0.7.
        min_samples (int, optional): The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. Defaults to 10.
        max_k (int, optional): Maximum number of clusters/subtracklets to be output by splitting component. Defaults to 3.
        min_len (int, optional): Minimum length for a tracklet required for splitting. Defaults to 100.
        spatial_factor (float, optional): Factor to adjust spatial distances. Defaults to 1.0.
        merge_dist_thres (float, optional): Minimum cosine distance between two tracklets for merging. Defaults to 0.4.
    """
    
    mot_files = sorted(list(mot_results_folder.glob("*.txt")))
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for mot_file in mot_files:
            seq_name = mot_file.stem
            dets_file = dets_folder / f"{seq_name}.txt"
            embs_file = embs_folder / f"{seq_name}.txt"
            
            if dets_file.exists() and embs_file.exists():
                futures.append(executor.submit(process_sequence, mot_file, dets_file, embs_file, 
                                               eps, min_samples, max_k, min_len, spatial_factor, merge_dist_thres))
            else:
                LOGGER.warning(f"Dets or Embs file not found for {seq_name}")

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="SCT Processing"):
            try:
                future.result()
            except Exception as e:
                LOGGER.error(f"Error in SCT: {e}")
                LOGGER.error(traceback.format_exc())
