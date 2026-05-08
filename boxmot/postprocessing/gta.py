# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

"""Online Global Tracklet Association (GTA) module.

Inspired by:
    Sun et al., "GTA: Global Tracklet Association for Multi-Object Tracking
    in Sports", ACCV 2024 Workshop.

The original GTA is an **offline** post-processing method that operates on
complete tracklets after a video is fully processed. This module provides an
**online** adaptation that can be plugged into any tracker as a per-frame
refinement step, addressing two types of errors:

1. **Mix-up errors** (Tracklet Splitter): A single track accumulates
   embeddings from multiple identities due to ID switches during occlusion.
   The online splitter periodically clusters recent embeddings and signals
   that the current track should be split (effectively re-born with a new ID)
   when a cluster break is detected.

2. **Cut-off errors** (Tracklet Connector): When a player re-enters the
   scene, the tracker assigns a new ID instead of re-using the old one.
   The online connector maintains a gallery of recently-lost tracklet
   features and attempts to re-link new tracklets to lost ones via cosine
   similarity with spatial constraints.

Usage
-----
    from boxmot.postprocessing.gta import OnlineGTA

    # Instantiate once (alongside the tracker)
    gta = OnlineGTA(
        splitter_enabled=True,
        connector_enabled=True,
    )

    # Each frame, after the tracker produces outputs:
    #   tracks: np.ndarray of shape (N, >=8) with columns
    #           [x1, y1, x2, y2, id, conf, cls, det_ind]
    #   lost_tracks: list of track objects that were just removed this frame
    #   active_tracks: list of currently active track objects
    #   (track objects must expose .id, .emb, .age, .history_observations)

    tracks = gta.refine(
        tracks=tracks,
        active_tracks=active_tracks,
        lost_tracks=lost_tracks,
        frame_shape=(H, W),
    )
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from boxmot.utils import logger as LOGGER


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class _TrackletGalleryEntry:
    """A cached representation of a lost tracklet for re-identification."""

    track_id: int
    mean_emb: np.ndarray  # L2-normalised average embedding
    emb_samples: np.ndarray  # (K, D) raw embedding samples for distance
    last_bbox: np.ndarray  # (4,) — last known [x1, y1, x2, y2]
    last_frame: int  # frame index when tracklet was lost
    cls: int  # object class


@dataclass
class _ActiveTrackState:
    """Per-track state maintained by the online splitter."""

    track_id: int
    emb_buffer: deque = field(default_factory=lambda: deque(maxlen=50))
    frame_indices: deque = field(default_factory=lambda: deque(maxlen=50))


# ---------------------------------------------------------------------------
# Online GTA Module
# ---------------------------------------------------------------------------


class OnlineGTA:
    """Online Global Tracklet Association — plug-and-play tracker refinement.

    Parameters
    ----------
    splitter_enabled : bool
        Enable the online tracklet splitter.
    splitter_interval : int
        Run the splitter every N frames per track (avoids per-frame overhead).
    splitter_min_samples : int
        DBSCAN ``min_samples`` — minimum points to form a cluster.
    splitter_eps : float
        DBSCAN ``eps`` — maximum cosine distance for neighbourhood.
    splitter_max_clusters : int
        If DBSCAN produces more clusters than this, merge smallest ones.
    splitter_window : int
        Number of recent embeddings to consider for splitting.
    connector_enabled : bool
        Enable the online tracklet connector (gallery-based re-ID).
    connector_gallery_ttl : int
        Maximum age (in frames) a lost tracklet stays in the gallery.
    connector_gallery_max : int
        Maximum number of entries in the gallery.
    connector_match_thresh : float
        Cosine distance threshold for accepting a gallery match (lower = stricter).
    connector_spatial_factor : float
        Fraction of frame diagonal used as maximum spatial gate between
        the lost tracklet's last position and the new tracklet's first position.
    connector_min_tracklet_len : int
        Minimum length (frames) of a lost tracklet before it is gallery-eligible.
    connector_max_young_age : int
        Maximum ``hit_streak`` of an active track to be considered "young"
        (i.e. a candidate for reconnection to the gallery). Tracks with
        more consecutive hits are considered established and skipped.
    """

    def __init__(
        self,
        # Splitter params
        splitter_enabled: bool = True,
        splitter_interval: int = 10,
        splitter_min_samples: int = 5,
        splitter_eps: float = 0.6,
        splitter_max_clusters: int = 3,
        splitter_window: int = 30,
        # Connector params
        connector_enabled: bool = True,
        connector_gallery_ttl: int = 150,
        connector_gallery_max: int = 200,
        connector_match_thresh: float = 0.4,
        connector_spatial_factor: float = 0.5,
        connector_min_tracklet_len: int = 5,
        connector_max_young_age: int = 8,
    ):
        # Splitter config
        self.splitter_enabled = splitter_enabled
        self.splitter_interval = max(splitter_interval, 1)
        self.splitter_min_samples = max(splitter_min_samples, 2)
        self.splitter_eps = splitter_eps
        self.splitter_max_clusters = max(splitter_max_clusters, 2)
        self.splitter_window = max(splitter_window, 10)

        # Connector config
        self.connector_enabled = connector_enabled
        self.connector_gallery_ttl = max(connector_gallery_ttl, 1)
        self.connector_gallery_max = max(connector_gallery_max, 1)
        self.connector_match_thresh = connector_match_thresh
        self.connector_spatial_factor = connector_spatial_factor
        self.connector_min_tracklet_len = max(connector_min_tracklet_len, 1)
        self.connector_max_young_age = max(connector_max_young_age, 1)

        # Internal state
        self._frame_count: int = 0
        self._track_states: dict[int, _ActiveTrackState] = {}
        self._gallery: list[_TrackletGalleryEntry] = []
        # Mapping: new_track_id -> reconnected_old_track_id
        self._id_remap: dict[int, int] = {}
        # Tracks flagged for splitting (set of track IDs)
        self._split_flags: set[int] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refine(
        self,
        tracks: np.ndarray,
        active_tracks: list[Any],
        lost_tracks: list[Any] | None = None,
        frame_shape: tuple[int, int] = (1080, 1920),
    ) -> np.ndarray:
        """Run the online GTA refinement for the current frame.

        Parameters
        ----------
        tracks : np.ndarray
            Tracker output for the current frame, shape (N, >=8).
            Columns: [x1, y1, x2, y2, id, conf, cls, det_ind].
        active_tracks : list
            Active track objects. Each must have attributes:
            ``id`` (int), ``emb`` (np.ndarray or None), ``age`` (int),
            ``history_observations`` (sequence of bboxes).
        lost_tracks : list or None
            Tracks that were removed/lost this frame. Same attribute
            requirements as active_tracks.
        frame_shape : tuple (H, W)
            Frame dimensions for spatial gating.

        Returns
        -------
        np.ndarray
            Refined tracks array (same shape). Track IDs may be remapped
            by the connector, or tracks may be flagged (via ``split_ids``
            property) for the caller to handle splitting.
        """
        self._frame_count += 1

        if tracks.size == 0:
            # Still process lost tracks for gallery
            if self.connector_enabled and lost_tracks:
                self._ingest_lost_tracks(lost_tracks)
            return tracks

        # --- Connector: ingest lost tracks into gallery ---
        if self.connector_enabled:
            if lost_tracks:
                self._ingest_lost_tracks(lost_tracks)
            # Attempt to reconnect young tracks to gallery entries
            tracks = self._connector_reconnect(tracks, active_tracks, frame_shape)
            # Expire old gallery entries
            self._expire_gallery()

        # --- Splitter: accumulate embeddings and check for splits ---
        if self.splitter_enabled:
            self._splitter_accumulate(active_tracks)
            if self._frame_count % self.splitter_interval == 0:
                self._splitter_check()

        return tracks

    @property
    def split_ids(self) -> set[int]:
        """Track IDs flagged for splitting by the online splitter.

        The calling tracker should:
        1. Read this set after calling ``refine()``.
        2. For each flagged track, assign a new ID to the track (effectively
           splitting the identity from this point onward).
        3. Call ``clear_split_flags()`` once handled.
        """
        return self._split_flags

    def clear_split_flags(self) -> None:
        """Clear the set of split-flagged track IDs."""
        self._split_flags.clear()

    @property
    def id_remap(self) -> dict[int, int]:
        """Mapping of new_track_id -> reconnected_old_track_id.

        The calling tracker can use this to remap IDs in its track objects.
        """
        return self._id_remap

    def clear_remap(self) -> None:
        """Clear the ID remap dict after the caller has applied remaps."""
        self._id_remap.clear()

    # ------------------------------------------------------------------
    # Connector internals
    # ------------------------------------------------------------------

    def _ingest_lost_tracks(self, lost_tracks: list[Any]) -> None:
        """Add qualifying lost tracks to the re-ID gallery."""
        for trk in lost_tracks:
            emb = getattr(trk, "emb", None)
            if emb is None:
                continue
            age = getattr(trk, "age", 0)
            if age < self.connector_min_tracklet_len:
                continue

            # Collect embedding samples from history if available
            features = getattr(trk, "features", None)
            if features is not None and len(features) > 0:
                emb_samples = np.array(list(features))
            else:
                emb_samples = emb.reshape(1, -1)

            # Last bbox
            hist = getattr(trk, "history_observations", None)
            if hist and len(hist) > 0:
                last_bbox = np.array(hist[-1][:4], dtype=float)
            else:
                last_bbox = np.zeros(4, dtype=float)

            cls_val = int(getattr(trk, "cls", 0))

            # Normalise mean embedding
            mean_emb = emb_samples.mean(axis=0)
            norm = np.linalg.norm(mean_emb)
            if norm > 0:
                mean_emb = mean_emb / norm

            entry = _TrackletGalleryEntry(
                track_id=trk.id,
                mean_emb=mean_emb,
                emb_samples=emb_samples,
                last_bbox=last_bbox,
                last_frame=self._frame_count,
                cls=cls_val,
            )
            self._gallery.append(entry)

        # Enforce gallery size limit (drop oldest)
        while len(self._gallery) > self.connector_gallery_max:
            self._gallery.pop(0)

    def _connector_reconnect(
        self,
        tracks: np.ndarray,
        active_tracks: list[Any],
        frame_shape: tuple[int, int],
    ) -> np.ndarray:
        """Attempt to reconnect young active tracks to gallery entries."""
        if not self._gallery:
            return tracks

        H, W = frame_shape
        spatial_gate = self.connector_spatial_factor * np.sqrt(H**2 + W**2)

        # Build a lookup from track id to active track object
        trk_by_id: dict[int, Any] = {}
        for trk in active_tracks:
            trk_by_id[trk.id] = trk

        # Identify young tracks (potential new-born tracklets to reconnect)
        # A track is considered "young" if its total age is low — this means
        # it was recently created. We do NOT use hit_streak because that
        # resets on any gap, causing recovered tracks to be falsely treated
        # as new-born (leading to incorrect reconnections).
        young_ids = []
        young_embs = []
        young_bboxes = []
        young_classes = []

        for trk in active_tracks:
            age = getattr(trk, "age", 0)
            if age <= self.connector_max_young_age and trk.id not in self._id_remap.values():
                emb = getattr(trk, "emb", None)
                if emb is None:
                    continue
                young_ids.append(trk.id)
                young_embs.append(emb)
                hist = getattr(trk, "history_observations", None)
                if hist and len(hist) > 0:
                    young_bboxes.append(np.array(hist[0][:4], dtype=float))
                else:
                    young_bboxes.append(np.zeros(4, dtype=float))
                young_classes.append(int(getattr(trk, "cls", 0)))

        if not young_ids:
            return tracks

        young_embs_arr = np.array(young_embs)  # (M, D)
        gallery_embs = np.array([g.mean_emb for g in self._gallery])  # (G, D)

        # Cosine distance matrix: (M, G)
        # cos_sim = young_embs_arr @ gallery_embs.T (both L2-normalised)
        norms_y = np.linalg.norm(young_embs_arr, axis=1, keepdims=True)
        norms_y = np.maximum(norms_y, 1e-8)
        young_normed = young_embs_arr / norms_y

        norms_g = np.linalg.norm(gallery_embs, axis=1, keepdims=True)
        norms_g = np.maximum(norms_g, 1e-8)
        gallery_normed = gallery_embs / norms_g

        cos_dist = 1.0 - young_normed @ gallery_normed.T  # (M, G)

        # Greedy matching: for each young track, find best gallery match
        matched_gallery_indices: set[int] = set()
        remap_this_frame: dict[int, int] = {}

        # Sort by minimum distance for greedy assignment
        for m_idx in range(len(young_ids)):
            best_g_idx = -1
            best_dist = self.connector_match_thresh

            for g_idx in range(len(self._gallery)):
                if g_idx in matched_gallery_indices:
                    continue

                # Class must match
                if self._gallery[g_idx].cls != young_classes[m_idx]:
                    continue

                dist = cos_dist[m_idx, g_idx]
                if dist >= best_dist:
                    continue

                # Spatial constraint: distance between gallery's last bbox
                # center and young track's first bbox center
                g_bbox = self._gallery[g_idx].last_bbox
                y_bbox = young_bboxes[m_idx]
                g_center = np.array([(g_bbox[0] + g_bbox[2]) / 2,
                                     (g_bbox[1] + g_bbox[3]) / 2])
                y_center = np.array([(y_bbox[0] + y_bbox[2]) / 2,
                                     (y_bbox[1] + y_bbox[3]) / 2])
                spatial_dist = np.linalg.norm(g_center - y_center)

                if spatial_dist > spatial_gate:
                    continue

                best_dist = dist
                best_g_idx = g_idx

            if best_g_idx >= 0:
                matched_gallery_indices.add(best_g_idx)
                old_id = self._gallery[best_g_idx].track_id
                new_id = young_ids[m_idx]
                remap_this_frame[new_id] = old_id

        # Apply remaps to output tracks array
        if remap_this_frame:
            id_col = 4  # standard output column for track ID
            for new_id, old_id in remap_this_frame.items():
                mask = tracks[:, id_col].astype(int) == new_id
                tracks[mask, id_col] = old_id
                self._id_remap[new_id] = old_id
                LOGGER.debug(
                    f"GTA Connector: remapped track {new_id} -> {old_id} "
                    f"(cosine dist={best_dist:.3f})"
                )

            # Remove matched entries from gallery
            self._gallery = [
                g for i, g in enumerate(self._gallery)
                if i not in matched_gallery_indices
            ]

        return tracks

    def _expire_gallery(self) -> None:
        """Remove gallery entries older than TTL."""
        cutoff = self._frame_count - self.connector_gallery_ttl
        self._gallery = [
            g for g in self._gallery if g.last_frame >= cutoff
        ]

    # ------------------------------------------------------------------
    # Splitter internals
    # ------------------------------------------------------------------

    def _splitter_accumulate(self, active_tracks: list[Any]) -> None:
        """Accumulate embeddings for active tracks."""
        seen_ids: set[int] = set()
        for trk in active_tracks:
            seen_ids.add(trk.id)
            emb = getattr(trk, "emb", None)
            if emb is None:
                continue
            state = self._track_states.get(trk.id)
            if state is None:
                state = _ActiveTrackState(
                    track_id=trk.id,
                    emb_buffer=deque(maxlen=self.splitter_window),
                    frame_indices=deque(maxlen=self.splitter_window),
                )
                self._track_states[trk.id] = state
            state.emb_buffer.append(emb.copy())
            state.frame_indices.append(self._frame_count)

        # Clean up states for tracks no longer active
        dead_ids = set(self._track_states.keys()) - seen_ids
        for tid in dead_ids:
            del self._track_states[tid]

    def _splitter_check(self) -> None:
        """Run DBSCAN on buffered embeddings to detect identity switches."""
        try:
            from sklearn.cluster import DBSCAN
        except ImportError:
            LOGGER.warning(
                "GTA Splitter requires scikit-learn. "
                "Install with: pip install scikit-learn"
            )
            self.splitter_enabled = False
            return

        for tid, state in list(self._track_states.items()):
            if len(state.emb_buffer) < self.splitter_min_samples * 2:
                continue  # not enough samples to split reliably

            embs = np.array(state.emb_buffer)  # (N, D)

            # Normalise
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            embs_normed = embs / norms

            # Cosine distance matrix (clamp to avoid negative values from fp noise)
            cos_sim = embs_normed @ embs_normed.T
            cos_dist_matrix = np.clip(1.0 - cos_sim, 0.0, 2.0)

            # DBSCAN with precomputed distance
            db = DBSCAN(
                eps=self.splitter_eps,
                min_samples=self.splitter_min_samples,
                metric="precomputed",
            )
            labels = db.fit_predict(cos_dist_matrix)

            # Assign outliers (-1) to nearest cluster
            unique_labels = set(labels)
            unique_labels.discard(-1)

            if len(unique_labels) <= 1:
                continue  # single identity — all good

            # Merge if too many clusters
            n_clusters = len(unique_labels)
            if n_clusters > self.splitter_max_clusters:
                # Keep only the largest clusters
                # (simplified: just check if there's a recent cluster break)
                pass

            # Check if the most recent samples belong to a different cluster
            # than the majority of older samples
            recent_n = min(self.splitter_min_samples, len(labels))
            recent_labels = labels[-recent_n:]
            older_labels = labels[:-recent_n]

            # If recent samples are predominantly in a different cluster,
            # flag for split
            if len(older_labels) == 0:
                continue

            # Most common label in older samples
            older_valid = older_labels[older_labels >= 0]
            if len(older_valid) == 0:
                continue
            majority_old = np.bincount(older_valid).argmax()

            # Most common label in recent samples
            recent_valid = recent_labels[recent_labels >= 0]
            if len(recent_valid) == 0:
                continue
            majority_recent = np.bincount(recent_valid).argmax()

            if majority_recent != majority_old:
                # Identity switch detected — flag for split
                self._split_flags.add(tid)
                LOGGER.debug(
                    f"GTA Splitter: flagged track {tid} for split "
                    f"(clusters: {n_clusters}, old_label={majority_old}, "
                    f"recent_label={majority_recent})"
                )
                # Reset the buffer to only keep recent samples
                recent_embs = list(state.emb_buffer)[-recent_n:]
                recent_frames = list(state.frame_indices)[-recent_n:]
                state.emb_buffer = deque(recent_embs, maxlen=self.splitter_window)
                state.frame_indices = deque(recent_frames, maxlen=self.splitter_window)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all internal state (e.g. between videos)."""
        self._frame_count = 0
        self._track_states.clear()
        self._gallery.clear()
        self._id_remap.clear()
        self._split_flags.clear()
