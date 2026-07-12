"""SAM2MOT – Hybrid bbox + mask tracker.

Uses externally provided segmentation masks (e.g., from Mask R-CNN) for
mask-IoU-based association, cross-object interaction (COI) occlusion handling,
three-stage matching, and frame-out recovery.

No SAM2 dependency – masks are supplied per-frame via the ``masks`` argument
of ``update()``.
"""

from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from boxmot.trackers.common.geometry.obb import xywha_to_xyxy
from boxmot.trackers.hybrid.base import HybridBaseTracker
from boxmot.utils import logger as LOGGER

# ---------------------------------------------------------------------------
# Track state labels
# ---------------------------------------------------------------------------

class TrackState:
    RELIABLE = "reliable"
    PENDING = "pending"
    SUSPICIOUS = "suspicious"
    LOST = "lost"
    FRAME_OUT = "frame_out"


# ---------------------------------------------------------------------------
# Internal track representation
# ---------------------------------------------------------------------------

@dataclass
class _Track:
    id: int
    bbox: np.ndarray
    mask: Optional[np.ndarray]
    confidence: float
    state: str
    lost_frames: int
    age: int
    conf_history: deque
    last_seen_frame: int
    init_frame: int
    prev_bbox: Optional[np.ndarray] = None
    velocity: Optional[np.ndarray] = None  # (4,) linear velocity [dx1, dy1, dx2, dy2]
    is_dense: bool = False
    last_matched_frame: Optional[int] = None
    last_matched_bbox: Optional[np.ndarray] = None
    last_matched_density: float = 0.0
    skip_memory_current: bool = False
    cls: int = 0
    det_ind: int = -1
    obb: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Trajectory Manager
# ---------------------------------------------------------------------------

class _TrajectoryManager:

    def __init__(self, tau_r: float, tau_p: float, tau_s: float, tolerance_frames: int,
                 untracked_ratio_threshold: float = 0.5):
        self.tau_r = tau_r
        self.tau_p = tau_p
        self.tau_s = tau_s
        self.tolerance_frames = tolerance_frames
        self.untracked_ratio_threshold = untracked_ratio_threshold

    def classify_state(self, confidence: float) -> str:
        if confidence > self.tau_r:
            return TrackState.RELIABLE
        elif confidence > self.tau_p:
            return TrackState.PENDING
        elif confidence > self.tau_s:
            return TrackState.SUSPICIOUS
        return TrackState.LOST

    def compute_untracked_mask(self, mask_shape: Tuple[int, int],
                               tracked_masks: List[np.ndarray],
                               guard_bboxes: List[np.ndarray],
                               scale: Tuple[float, ...] = (1.0, 0.0, 0.0)) -> np.ndarray:
        """Compute untracked region at mask resolution.

        Args:
            mask_shape: (mH, mW) resolution of the masks.
            tracked_masks: list of existing track masks at mask resolution.
            guard_bboxes: bboxes in image coordinates.
            scale: (scale_factor, pad_y, pad_x) letterbox-aware transform.
        """
        mH, mW = mask_shape
        s, pad_y, pad_x = scale if len(scale) == 3 else (scale[0], 0.0, 0.0)
        untracked = np.ones((mH, mW), dtype=np.uint8)
        for m in tracked_masks:
            if m is not None and m.shape == (mH, mW):
                untracked[m > 0] = 0
        for bbox in guard_bboxes:
            if bbox is not None:
                x1 = max(0, int(bbox[0] * s + pad_x))
                y1 = max(0, int(bbox[1] * s + pad_y))
                x2 = min(mW, int(bbox[2] * s + pad_x))
                y2 = min(mH, int(bbox[3] * s + pad_y))
                if x2 > x1 and y2 > y1:
                    untracked[y1:y2, x1:x2] = 0
        return untracked

    def should_add_detection(self, bbox: np.ndarray, untracked_mask: np.ndarray,
                             scale: Tuple[float, ...] = (1.0, 0.0, 0.0)) -> bool:
        s, pad_y, pad_x = scale if len(scale) == 3 else (scale[0], 0.0, 0.0)
        H, W = untracked_mask.shape
        x1 = max(0, int(bbox[0] * s + pad_x))
        y1 = max(0, int(bbox[1] * s + pad_y))
        x2 = min(W, int(bbox[2] * s + pad_x))
        y2 = min(H, int(bbox[3] * s + pad_y))
        area = (x2 - x1) * (y2 - y1)
        if area <= 0:
            return False
        overlap = untracked_mask[y1:y2, x1:x2].sum()
        return (overlap / area) > self.untracked_ratio_threshold

    def should_remove(self, track: _Track) -> bool:
        return track.lost_frames > self.tolerance_frames


# ---------------------------------------------------------------------------
# Cross-Object Interaction
# ---------------------------------------------------------------------------

class _CrossObjectInteraction:

    def __init__(self, miou_threshold: float = 0.8, variance_history: int = 10):
        self.miou_threshold = miou_threshold
        self.variance_history = variance_history

    @staticmethod
    def mask_iou(m1: np.ndarray, m2: np.ndarray) -> float:
        if m1 is None or m2 is None:
            return 0.0
        if m1.shape != m2.shape:
            return 0.0
        inter = np.logical_and(m1, m2).sum()
        union = np.logical_or(m1, m2).sum()
        return float(inter) / max(float(union), 1e-6)

    def _mean_conf(self, history: deque) -> float:
        if len(history) < 2:
            return 0.0
        vals = list(history)[-self.variance_history:]
        return float(np.mean(vals))

    def _var_conf(self, history: deque) -> float:
        if len(history) < 2:
            return 0.0
        vals = list(history)[-self.variance_history:]
        return float(np.var(vals))

    def detect_and_resolve(self, tracks: List[_Track]) -> List[int]:
        """Return IDs of tracks that should skip memory (occluded).

        Uses bbox overlap as a pre-filter to avoid expensive mask IoU on
        all O(n²) pairs.
        """
        skip_ids = []
        n = len(tracks)
        # Only check pairs whose bboxes overlap (necessary condition for mask overlap)
        for i in range(n):
            a = tracks[i]
            if a.mask is None or a.state == TrackState.FRAME_OUT:
                continue
            for j in range(i + 1, n):
                b = tracks[j]
                if b.mask is None or b.state == TrackState.FRAME_OUT:
                    continue
                # Quick bbox overlap check — skip expensive mask IoU if no bbox overlap
                if not self._bboxes_overlap(a.bbox, b.bbox):
                    continue
                miou = self.mask_iou(a.mask, b.mask)
                if miou <= self.miou_threshold:
                    continue
                mean_a = self._mean_conf(a.conf_history)
                mean_b = self._mean_conf(b.conf_history)
                var_a = self._var_conf(a.conf_history)
                var_b = self._var_conf(b.conf_history)
                diff_mean = abs(mean_a - mean_b)
                diff_var = abs(var_a - var_b)
                if diff_mean >= diff_var:
                    occluded = a if mean_a < mean_b else b
                else:
                    occluded = a if var_a > var_b else b
                occluded.skip_memory_current = True
                if occluded.id not in skip_ids:
                    skip_ids.append(occluded.id)
        return skip_ids

    @staticmethod
    def _bboxes_overlap(a: np.ndarray, b: np.ndarray) -> bool:
        """Fast check if two bboxes overlap at all."""
        return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


# ---------------------------------------------------------------------------
# Sam2Mot tracker (mask-based, no SAM2 dependency)
# ---------------------------------------------------------------------------

class Sam2Mot(HybridBaseTracker):
    """Hybrid bbox + mask tracker with three-stage matching, COI, and frame-out recovery.

    This tracker uses externally provided segmentation masks for mask-IoU-based
    association. Despite the name (for continuity with the paper), it does **not**
    require SAM2 – any source of per-detection masks works (Mask R-CNN, etc.).
    """

    supports_masks = True
    supports_obb = True

    def __init__(
        self,
        # Base tracker params
        det_thresh: float = 0.3,
        max_age: int = 60,
        min_hits: int = 1,
        iou_threshold: float = 0.3,
        per_class: bool = False,
        # Sam2Mot-specific params
        tolerance_frames: int = 30,
        memory_window: int = 25,
        cost_weight: float = 0.5,
        tau_r: float = 0.8,
        tau_p: float = 0.5,
        tau_s: float = 0.3,
        density_threshold: float = 0.9,
        second_stage_iou_threshold: float = 0.3,
        frame_out_d_thre: float = 0.6,
        miou_threshold: float = 0.8,
        untracked_ratio_threshold: float = 0.5,
        new_track_thresh: float = 0.5,
        **kwargs,
    ):
        super().__init__(
            det_thresh=det_thresh,
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold,
            per_class=per_class,
        )
        self.tolerance_frames = tolerance_frames
        self.memory_window = memory_window
        self.cost_weight = cost_weight
        self.density_threshold = density_threshold
        self.second_stage_iou_threshold = second_stage_iou_threshold
        self.frame_out_d_thre = frame_out_d_thre
        self.new_track_thresh = new_track_thresh

        self.trajectory_manager = _TrajectoryManager(
            tau_r=tau_r, tau_p=tau_p, tau_s=tau_s,
            tolerance_frames=tolerance_frames,
            untracked_ratio_threshold=untracked_ratio_threshold,
        )
        self.coi = _CrossObjectInteraction(miou_threshold=miou_threshold)

        # Internal state
        self._tracks: List[_Track] = []
        self._next_id = 1
        self._frame_count = 0

        LOGGER.info(
            f"Sam2Mot: det_thresh={det_thresh}, tolerance_frames={tolerance_frames}, "
            f"cost_weight={cost_weight}, density_threshold={density_threshold}, "
            f"miou_threshold={miou_threshold}"
        )

    def reset(self):
        """Reset tracker state."""
        self._tracks = []
        self._next_id = 1
        self._frame_count = 0

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def _update_impl(self, dets: np.ndarray, img: np.ndarray,
                     embs: np.ndarray = None, masks: np.ndarray = None):
        """Process one frame.

        Args:
            dets: (N, 6) detections [x1, y1, x2, y2, conf, cls].
            img: Current frame (H, W, 3).
            embs: Ignored (no ReID).
            masks: (N, H, W) binary masks aligned to dets.

        Returns:
            Tuple of (tracks_array, output_masks):
                tracks_array: (M, 8) [x1, y1, x2, y2, id, conf, cls, det_ind]
                output_masks: (M, H, W) or None
        """
        self._frame_count += 1
        frame_id = self._frame_count
        H, W = img.shape[:2]

        # Mask operations use enclosing AABBs while OBB geometry is preserved for output.
        det_obbs = dets[:, :5].copy() if self.is_obb and len(dets) else None
        det_bboxes = (
            xywha_to_xyxy(det_obbs) if det_obbs is not None
            else dets[:, :4] if len(dets) else np.zeros((0, 4))
        )
        conf_col = 5 if self.is_obb else 4
        cls_col = 6 if self.is_obb else 5
        det_confs = dets[:, conf_col] if len(dets) > 0 else np.zeros(0)
        det_classes = dets[:, cls_col].astype(int) if len(dets) > 0 else np.zeros(0, dtype=int)
        n_dets = len(dets)

        # Masks array (may be at a different resolution than the image)
        det_masks = masks if (masks is not None and len(masks) == n_dets) else None
        if det_masks is not None:
            mH, mW = det_masks.shape[1], det_masks.shape[2]
        else:
            mH, mW = H, W
        # Letterbox-aware scale factors: image coords -> mask coords
        # Masks are in letterboxed model space (square with padding), not proportional to image
        scale = min(mH / H, mW / W)
        self._mask_scale = scale
        self._mask_pad_x = (mW - int(W * scale)) / 2.0
        self._mask_pad_y = (mH - int(H * scale)) / 2.0

        # Update existing track states
        for track in self._tracks:
            track.prev_bbox = track.bbox.copy() if track.bbox is not None else None
            track.age += 1

        active_tracks = [t for t in self._tracks if t.state != TrackState.LOST]

        # --- Identify frame-out candidates ---
        # Only move to frame-out after a long gap (10+ frames unmatched)
        frame_out_tracks = []
        normal_tracks = []
        for t in active_tracks:
            if (t.last_matched_frame is not None
                    and t.last_matched_frame <= frame_id - 10
                    and not t.is_dense
                    and t.age > 1):
                t.state = TrackState.FRAME_OUT
                t.mask = None
                frame_out_tracks.append(t)
            else:
                normal_tracks.append(t)

        # === Stage 1+2: Two-stage matching on normal tracks ===
        all_matches, unmatched_dets, unmatched_trk_indices, second_stage_matches = \
            self._two_stage_matching(det_bboxes, det_confs, normal_tracks, det_masks=det_masks)

        # Apply matches
        matched_track_ids = set()
        tracks_need_reconstruction = []

        for det_idx, trk_idx in all_matches:
            track = normal_tracks[trk_idx]
            bbox = det_bboxes[det_idx]
            conf = det_confs[det_idx]
            density = self._compute_density(det_idx, det_bboxes)

            track.last_matched_density = density
            track.is_dense = density > self.frame_out_d_thre
            track.last_matched_frame = frame_id
            track.last_matched_bbox = bbox.copy()
            matched_track_ids.add(track.id)

            is_second_stage = (det_idx, trk_idx) in set(second_stage_matches)

            if is_second_stage:
                if density >= self.density_threshold:
                    # Skip reconstruction for dense second-stage
                    pass
                else:
                    tracks_need_reconstruction.append((track, det_idx))
            else:
                # Crop mask to detection bbox region (in mask coordinates)
                if track.mask is not None and det_masks is not None and det_idx < len(det_masks):
                    x1 = max(0, int(bbox[0] * self._mask_scale + self._mask_pad_x))
                    y1 = max(0, int(bbox[1] * self._mask_scale + self._mask_pad_y))
                    x2 = min(mW, int(bbox[2] * self._mask_scale + self._mask_pad_x))
                    y2 = min(mH, int(bbox[3] * self._mask_scale + self._mask_pad_y))
                    cropped = np.zeros_like(track.mask)
                    cropped[y1:y2, x1:x2] = track.mask[y1:y2, x1:x2]
                    track.mask = cropped

                # Check if quality reconstruction needed
                if track.state == TrackState.PENDING and conf > self.trajectory_manager.tau_r:
                    if density < self.density_threshold:
                        tracks_need_reconstruction.append((track, det_idx))

            # Update velocity (exponential moving average)
            new_vel = bbox - track.bbox
            if track.velocity is not None:
                track.velocity = 0.6 * track.velocity + 0.4 * new_vel
            else:
                track.velocity = new_vel

            track.bbox = bbox.copy()
            track.obb = det_obbs[det_idx].copy() if det_obbs is not None else None
            track.confidence = conf
            track.conf_history.append(conf)
            track.last_seen_frame = frame_id
            track.lost_frames = 0
            track.cls = det_classes[det_idx]
            track.det_ind = det_idx

            # Assign mask from detection
            if det_masks is not None and det_idx < len(det_masks):
                track.mask = det_masks[det_idx]

            # Update state
            new_state = self.trajectory_manager.classify_state(conf)
            if new_state != TrackState.LOST:
                track.state = new_state

        # --- Cross-Object Interaction ---
        if len(active_tracks) > 1:
            coi_skip_ids = self.coi.detect_and_resolve(active_tracks)
            for track in active_tracks:
                if track.id in coi_skip_ids and track.skip_memory_current:
                    track.mask = None
                    track.skip_memory_current = False

        # Reconstruct tracks that need it
        for track, det_idx in tracks_need_reconstruction:
            if det_masks is not None and det_idx < len(det_masks):
                track.mask = det_masks[det_idx]
            track.state = TrackState.RELIABLE
            track.bbox = det_bboxes[det_idx].copy()
            track.obb = det_obbs[det_idx].copy() if det_obbs is not None else None
            track.confidence = det_confs[det_idx]
            track.conf_history.append(det_confs[det_idx])
            track.det_ind = det_idx

        # Increment lost frames for unmatched tracks
        for t in self._tracks:
            if t.id not in matched_track_ids:
                t.lost_frames += 1
                if t.lost_frames > self.trajectory_manager.tolerance_frames:
                    t.state = TrackState.LOST

        # === Stage 3: Frame-out recovery ===
        if frame_out_tracks and unmatched_dets:
            fo_matches = self._frame_out_matching(
                det_bboxes, unmatched_dets, frame_out_tracks
            )
            for det_idx, fo_track in fo_matches:
                bbox = det_bboxes[det_idx]
                conf = det_confs[det_idx]
                density = self._compute_density(det_idx, det_bboxes)
                fo_track.state = TrackState.RELIABLE
                fo_track.bbox = bbox.copy()
                fo_track.obb = det_obbs[det_idx].copy() if det_obbs is not None else None
                fo_track.confidence = conf
                fo_track.conf_history.append(conf)
                fo_track.last_seen_frame = frame_id
                fo_track.lost_frames = 0
                fo_track.last_matched_frame = frame_id
                fo_track.last_matched_bbox = bbox.copy()
                fo_track.last_matched_density = density
                fo_track.is_dense = density > self.frame_out_d_thre
                fo_track.cls = det_classes[det_idx]
                fo_track.det_ind = det_idx
                if det_masks is not None and det_idx < len(det_masks):
                    fo_track.mask = det_masks[det_idx]
                matched_track_ids.add(fo_track.id)
                unmatched_dets = [d for d in unmatched_dets if d != det_idx]

        # === Add new tracks for unmatched detections ===
        if unmatched_dets:
            tracked_masks_list = [t.mask for t in self._tracks
                                  if t.mask is not None and t.state != TrackState.LOST]
            guard_bboxes = []
            for t in active_tracks:
                if t.mask is None or not np.any(t.mask):
                    gb = t.last_matched_bbox if t.last_matched_bbox is not None else t.bbox
                    if gb is not None:
                        guard_bboxes.append(gb)
                elif t.is_dense and t.last_matched_bbox is not None:
                    guard_bboxes.append(t.last_matched_bbox)

            untracked = self.trajectory_manager.compute_untracked_mask(
                (mH, mW), tracked_masks_list, guard_bboxes,
                scale=(self._mask_scale, self._mask_pad_y, self._mask_pad_x),
            )

            for det_idx in unmatched_dets:
                bbox = det_bboxes[det_idx]
                conf = det_confs[det_idx]
                # Only create new tracks from high-confidence detections
                if conf < self.new_track_thresh:
                    continue
                if not self.trajectory_manager.should_add_detection(
                    bbox, untracked, scale=(self._mask_scale, self._mask_pad_y, self._mask_pad_x)
                ):
                    continue

                density = self._compute_density(det_idx, det_bboxes)
                mask = det_masks[det_idx] if (det_masks is not None and det_idx < len(det_masks)) else None
                new_track = _Track(
                    id=self._next_id,
                    bbox=bbox.copy(),
                    mask=mask,
                    confidence=conf,
                    state=TrackState.RELIABLE,
                    lost_frames=0,
                    age=1,
                    conf_history=deque(maxlen=self.memory_window),
                    last_seen_frame=frame_id,
                    init_frame=frame_id,
                    last_matched_frame=frame_id,
                    last_matched_bbox=bbox.copy(),
                    last_matched_density=density,
                    is_dense=density > self.frame_out_d_thre,
                    cls=det_classes[det_idx],
                    det_ind=det_idx,
                    obb=det_obbs[det_idx].copy() if det_obbs is not None else None,
                )
                new_track.conf_history.append(conf)
                self._tracks.append(new_track)
                matched_track_ids.add(self._next_id)
                self._next_id += 1

        # === Remove dead tracks ===
        self._tracks = [t for t in self._tracks if not self.trajectory_manager.should_remove(t)]

        # === Build output ===
        output_tracks = []
        output_masks_list = []

        for track in self._tracks:
            if track.id not in matched_track_ids:
                continue
            if track.age < self.min_hits and self._frame_count > self.min_hits:
                continue
            box = track.obb if self.is_obb else track.bbox
            row = np.array([
                *box,
                track.id, track.confidence, track.cls, track.det_ind,
            ], dtype=np.float64)
            output_tracks.append(row)
            output_masks_list.append(track.mask)

        if output_tracks:
            tracks_array = np.array(output_tracks)
            # Build output masks at mask resolution (not image resolution)
            has_any_mask = any(m is not None and m.shape == (mH, mW) and np.any(m) for m in output_masks_list)
            if has_any_mask:
                out_masks = np.zeros((len(output_masks_list), mH, mW), dtype=np.uint8)
                for i, m in enumerate(output_masks_list):
                    if m is not None and m.shape == (mH, mW):
                        out_masks[i] = m
                return tracks_array, out_masks
            return tracks_array, None
        else:
            return np.empty((0, 9 if self.is_obb else 8)), None

    # ------------------------------------------------------------------
    # Matching helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _iou_matrix(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
        """Compute IoU matrix between two sets of bboxes (vectorized).

        Args:
            bboxes_a: (M, 4) array [x1, y1, x2, y2]
            bboxes_b: (N, 4) array [x1, y1, x2, y2]
        Returns:
            (M, N) IoU matrix
        """
        # Broadcast: (M, 1, 4) vs (1, N, 4)
        a = bboxes_a[:, None, :]  # (M, 1, 4)
        b = bboxes_b[None, :, :]  # (1, N, 4)
        ix1 = np.maximum(a[..., 0], b[..., 0])
        iy1 = np.maximum(a[..., 1], b[..., 1])
        ix2 = np.minimum(a[..., 2], b[..., 2])
        iy2 = np.minimum(a[..., 3], b[..., 3])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
        area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
        union = area_a + area_b - inter
        return inter / np.maximum(union, 1e-6)

    def _two_stage_matching(self, det_bboxes: np.ndarray, det_confs: np.ndarray,
                            tracks: List[_Track], det_masks=None):
        """Two-stage matching: high-conf first, then low-conf on remaining tracks."""
        n_dets = len(det_bboxes)
        n_trks = len(tracks)

        if n_dets == 0 or n_trks == 0:
            return [], list(range(n_dets)), list(range(n_trks)), []

        # Motion-predicted bboxes for tracks
        trk_bboxes = np.array([
            t.bbox + t.velocity if t.velocity is not None else t.bbox
            for t in tracks
        ])

        # Split detections into high and low confidence
        high_conf_mask = det_confs >= self.det_thresh
        high_inds = np.where(high_conf_mask)[0]
        low_inds = np.where(~high_conf_mask)[0]

        matches_all = []
        matched_dets = set()
        matched_trks = set()

        # --- Pass 1: Match high-confidence detections ---
        if len(high_inds) > 0:
            iou_mat = self._iou_matrix(det_bboxes[high_inds], trk_bboxes)
            cost = np.where(iou_mat > 0, 1.0 - iou_mat, 1.0)

            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < 1.0:
                    orig_det = high_inds[r]
                    matches_all.append((int(orig_det), c))
                    matched_dets.add(int(orig_det))
                    matched_trks.add(c)

        # --- Pass 2: Match low-confidence detections to remaining tracks ---
        unmatched_trks_pass1 = [j for j in range(n_trks) if j not in matched_trks]
        if len(low_inds) > 0 and unmatched_trks_pass1:
            remain_trk_bboxes = trk_bboxes[unmatched_trks_pass1]
            iou_mat2 = self._iou_matrix(det_bboxes[low_inds], remain_trk_bboxes)
            cost2 = np.where(iou_mat2 > 0, 1.0 - iou_mat2, 1.0)

            r2, c2 = linear_sum_assignment(cost2)
            for ri, ci in zip(r2, c2):
                if iou_mat2[ri, ci] > 0.3:  # require reasonable overlap for low-conf
                    orig_det = low_inds[ri]
                    orig_trk = unmatched_trks_pass1[ci]
                    matches_all.append((int(orig_det), orig_trk))
                    matched_dets.add(int(orig_det))
                    matched_trks.add(orig_trk)

        unmatched_dets = [i for i in range(n_dets) if i not in matched_dets]
        unmatched_trks = [j for j in range(n_trks) if j not in matched_trks]

        # Stage 2: last_matched_bbox for still-unmatched tracks (recovery)
        second_stage_matches = []
        if unmatched_dets and unmatched_trks:
            valid_trks = [(idx, tracks[idx]) for idx in unmatched_trks
                          if tracks[idx].last_matched_bbox is not None]
            if valid_trks:
                um_dets_arr = det_bboxes[unmatched_dets]
                trk_last_bboxes = np.array([trk.last_matched_bbox for _, trk in valid_trks])
                iou2 = self._iou_matrix(um_dets_arr, trk_last_bboxes)
                cost2 = np.where(iou2 > 0, 1.0 - iou2, 1.0)

                r2, c2 = linear_sum_assignment(cost2)
                matched_dets_s2 = set()
                matched_trks_s2 = set()
                for ri, ci in zip(r2, c2):
                    if cost2[ri, ci] < 1.0:
                        orig_det = unmatched_dets[ri]
                        orig_trk = valid_trks[ci][0]
                        iou = 1.0 - cost2[ri, ci]
                        if iou > self.second_stage_iou_threshold:
                            second_stage_matches.append((orig_det, orig_trk))
                            matched_dets_s2.add(orig_det)
                            matched_trks_s2.add(orig_trk)

                unmatched_dets = [d for d in unmatched_dets if d not in matched_dets_s2]
                unmatched_trks = [t for t in unmatched_trks if t not in matched_trks_s2]

        all_matches = matches_all + second_stage_matches
        return all_matches, unmatched_dets, unmatched_trks, second_stage_matches

    def _frame_out_matching(self, det_bboxes: np.ndarray,
                            unmatched_dets: List[int],
                            frame_out_tracks: List[_Track]) -> List[Tuple[int, _Track]]:
        """Stage 3: Match unmatched detections to frame-out tracks."""
        if not unmatched_dets or not frame_out_tracks:
            return []

        um_bboxes = det_bboxes[unmatched_dets]  # (K, 4)
        fo_bboxes = np.array([
            trk.last_matched_bbox if trk.last_matched_bbox is not None
            else np.zeros(4)
            for trk in frame_out_tracks
        ])
        has_bbox = np.array([trk.last_matched_bbox is not None for trk in frame_out_tracks])
        iou_mat = self._iou_matrix(um_bboxes, fo_bboxes)
        # Mask out tracks without a last_matched_bbox
        iou_mat[:, ~has_bbox] = 0
        cost = np.where(iou_mat > 0, 1.0 - iou_mat, 1.0)

        row_ind, col_ind = linear_sum_assignment(cost)
        results = []
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < 1.0:
                results.append((unmatched_dets[r], frame_out_tracks[c]))
        return results

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return float(inter) / max(float(union), 1e-6)

    def _compute_density(self, target_idx: int, all_bboxes: np.ndarray) -> float:
        """Compute overlap density for a detection relative to all others (vectorized)."""
        bbox = all_bboxes[target_idx]
        x1, y1, x2, y2 = bbox
        area = max((x2 - x1) * (y2 - y1), 1e-6)
        # Vectorized intersection computation
        ix1 = np.maximum(x1, all_bboxes[:, 0])
        iy1 = np.maximum(y1, all_bboxes[:, 1])
        ix2 = np.minimum(x2, all_bboxes[:, 2])
        iy2 = np.minimum(y2, all_bboxes[:, 3])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        inter[target_idx] = 0  # exclude self
        return float(inter.sum() / area)
