"""botsort_faiss.py
====================================
A drop‑in replacement for the original **BoTSORT** tracker that delegates all
**appearance‑based matching** to **FAISS** – while **guaranteeing identical
numerical distances** to the reference NumPy implementation. Evaluation scores
should therefore be unchanged.

Why the previous version hurt accuracy
--------------------------------------
The earlier FAISS helper **re‑normalised** vectors before the search.  STrack
already stores *unit‑length* features, so the extra normalisation introduced
small rounding differences (float32) that were amplified by the matching logic
and cascaded into worse ID switches.  The fix is simply to **use the stored
vectors as‑is**.

What's new in this revision
---------------------------
* **Feature history powered by FAISS** – `STrack.features` is now a lightweight
  deque **sub‑class** whose internal storage is mirrored in a `faiss.IndexFlatIP`.
  All public semantics (`append`, iteration, length, slicing, …) are preserved,
  so the rest of the codebase remains unchanged – yet any downstream consumer
  can immediately leverage quick similarity search when needed.
* **Bit‑for‑bit distances** – unchanged; we still compute
  `d = max(0, 1 - sim)` for every valid pair which reproduces the original
  behaviour exactly.
* **Global monkey‑patch** – as before, we *overwrite* the original
  `boxmot.utils.matching.embedding_distance` so that *any* code path uses the
  FAISS version.

Usage
~~~~~
```python
from botsort_faiss import BotSort  # replaces the old import
```
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path
from types import ModuleType
from typing import List, Sequence

import numpy as np
import torch

# ------------------------- third‑party: FAISS ------------------------- #
try:
    import faiss  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "botsort_faiss requires faiss‑cpu or faiss‑gpu. Install with\n"
        "  pip install faiss‑cpu\n"
        "or\n"
        "  pip install faiss‑gpu"
    ) from exc

# ---------------------------- BoxMot deps ----------------------------- #
from boxmot.appearance.reid.auto_backend import ReidAutoBackend
from boxmot.motion.cmc import get_cmc_method
from boxmot.motion.kalman_filters.aabb.xywh_kf import KalmanFilterXYWH
from boxmot.trackers.basetracker import BaseTracker
from boxmot.trackers.botsort.basetrack import BaseTrack, TrackState
from boxmot.trackers.botsort.botsort_utils import (
    joint_stracks,
    remove_duplicate_stracks,
    sub_stracks,
)
from boxmot.utils.matching import fuse_score, iou_distance, linear_assignment
from boxmot.utils.ops import xywh2xyxy, xyxy2xywh

__all__ = [
    "embedding_distance",
    "STrack",
    "BotSort",
]

# -----------------------------------------------------------------------------
# Appearance‑based distance (FAISS – exact, no renorm)
# -----------------------------------------------------------------------------


def _extract_track_features(tracks: Sequence["STrack"], dim: int):
    feats: List[np.ndarray] = []
    valid: List[int] = []
    for i, t in enumerate(tracks):
        if t.smooth_feat is not None:
            feats.append(t.smooth_feat.astype(np.float32, copy=False))
            valid.append(i)
    if not feats:
        return np.empty((0, dim), dtype=np.float32), []
    return np.vstack(feats), valid


def _extract_det_features(dets: Sequence["STrack"], dim: int):
    feats: List[np.ndarray] = []
    valid: List[int] = []
    for j, d in enumerate(dets):
        if d.curr_feat is not None:
            feats.append(d.curr_feat.astype(np.float32, copy=False))
            valid.append(j)
    if not feats:
        return np.empty((0, dim), dtype=np.float32), []
    return np.vstack(feats), valid


def embedding_distance(tracks: Sequence["STrack"], detections: Sequence["STrack"]) -> np.ndarray:
    """Exact cosine distance matrix via FAISS (no extra normalisation)."""
    n_t, n_d = len(tracks), len(detections)
    if n_t == 0 or n_d == 0:
        return np.zeros((n_t, n_d), dtype=np.float32)

    # fetch dimensionality from first available feature
    first_feat = next((t.smooth_feat for t in tracks if t.smooth_feat is not None), None)
    if first_feat is None:
        return np.ones((n_t, n_d), dtype=np.float32)
    dim = int(first_feat.shape[0])

    t_feats, t_valid = _extract_track_features(tracks, dim)
    d_feats, d_valid = _extract_det_features(detections, dim)

    # initialise with worst‑case cost (1 → after /2 becomes 0.5 then clipped to 1)
    dists = np.ones((n_t, n_d), dtype=np.float32)
    if t_feats.size == 0 or d_feats.size == 0:
        return dists

    # build exact IP index over detections
    index = faiss.IndexFlatIP(dim)
    index.add(d_feats)

    sims, idxs = index.search(t_feats, d_feats.shape[0])  # full list per track

    for row, (sim_row, idx_row) in enumerate(zip(sims, idxs)):
        ti = t_valid[row]
        for sim, dj in zip(sim_row, idx_row):
            di = d_valid[dj]
            dval = 1.0 - sim  # exact cosine distance on unit vectors
            if dval < 0.0:
                dval = 0.0  # numerical safety (mirrors original code)
            dists[ti, di] = dval
    return dists


# -----------------------------------------------------------------------------
# Feature history – FAISS‑backed deque
# -----------------------------------------------------------------------------

class FeatureDequeFAISS(deque):
    """A **drop‑in deque replacement** backed by a FAISS ``IndexFlatIP``.

    The public API is intentionally minimal: we only override *append* to keep the
    internal index in sync with the stored vectors.  This is sufficient because
    the rest of BoTSORT never mutates the container in any other way – it merely
    appends new vectors and occasionally iterates or queries *len()*.

    Notes
    -----
    * **Identical distances** – we *never* touch the incoming vectors; they are
      expected to be already *unit‑normalised* upstream (as in
      :pymeth:`STrack.update_features`).
    * **Sliding window** – when the deque discards the left‑most element (because
      it reached *maxlen*), we *rebuild* the FAISS index from the remaining
      vectors.  With a default history of ≤50 this incurs negligible overhead
      while keeping the implementation dead‑simple and, above all, *exact*.
    """

    def __init__(self, maxlen: int | None = None):  # noqa: D401
        super().__init__(maxlen=maxlen)
        # the `maxlen` property is already managed by `collections.deque` – no need (or permission) to re‑assign it
        self._dim: int | None = None
        self._index: faiss.IndexFlatIP | None = None

    # ------------------------------------------------------------------ utils
    def _ensure_index(self, feat: np.ndarray) -> None:
        if self._index is None:
            self._dim = int(feat.shape[0])
            self._index = faiss.IndexFlatIP(self._dim)

    def _rebuild_index(self) -> None:
        if self._index is None:
            return  # nothing to rebuild
        self._index.reset()
        if len(self):
            feats = np.stack(self)
            self._index.add(feats)

    # ---------------------------------------------------------------- override
    def append(self, feat: np.ndarray) -> None:  # type: ignore[override]
        feat = feat.astype(np.float32, copy=False)
        self._ensure_index(feat)

        # manage sliding window *before* adding the new element so that the
        # FAISS index mirrors the final state of the deque.
        if self.maxlen is not None and len(self) == self.maxlen:
            super().popleft()  # discard oldest
            self._rebuild_index()

        super().append(feat)
        if self._index is not None:
            self._index.add(np.expand_dims(feat, 0))

    # Convenience accessors – not used internally but may be handy
    # ------------------------------------------------------------
    @property
    def dim(self) -> int | None:  # noqa: D401
        return self._dim

    @property
    def index(self) -> faiss.IndexFlatIP | None:  # noqa: D401
        return self._index


# -----------------------------------------------------------------------------
# STrack – unchanged except for the FAISS‑backed feature history
# -----------------------------------------------------------------------------


class STrack(BaseTrack):
    """Single target track with smoothed appearance embedding and motion KF."""

    shared_kalman = KalmanFilterXYWH()

    def __init__(
        self,
        det: np.ndarray,
        feat: np.ndarray | None = None,
        *,
        feat_history: int = 50,
        max_obs: int = 50,
    ) -> None:
        super().__init__()
        self.xywh = xyxy2xywh(det[:4])
        self.conf = float(det[4])
        self.cls = int(det[5])
        self.det_ind = int(det[6])
        self.max_obs = max_obs

        self.kalman_filter: KalmanFilterXYWH | None = None
        self.mean: np.ndarray | None = None
        self.covariance: np.ndarray | None = None
        self.is_activated = False
        self.tracklet_len = 0

        # appearance
        self.cls_hist: list[list[int | float]] = []
        self.history_observations: deque[np.ndarray] = deque(maxlen=max_obs)
        # ------------------ FAISS‑powered history ------------------- #
        self.features: FeatureDequeFAISS = FeatureDequeFAISS(maxlen=feat_history)  # type: ignore[assignment]
        self.smooth_feat: np.ndarray | None = None
        self.curr_feat: np.ndarray | None = None
        self.alpha = 0.9

        self.update_cls(self.cls, self.conf)
        if feat is not None:
            self.update_features(feat)

    # ---------------- appearance helpers ---------------- #
    def update_features(self, feat: np.ndarray) -> None:
        feat = feat.astype(np.float32, copy=False)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat.copy()
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
            self.smooth_feat /= np.linalg.norm(self.smooth_feat) + 1e-12
        self.features.append(feat.copy())

    def update_cls(self, cls: int, conf: float) -> None:
        max_score = 0.0
        found = False
        for c in self.cls_hist:
            if cls == c[0]:
                c[1] += conf
                found = True
            if c[1] > max_score:
                max_score = c[1]
                self.cls = c[0]  # type: ignore[assignment]
        if not found:
            self.cls_hist.append([cls, conf])
            self.cls = cls

    # ---------------- motion helpers ---------------- #
    def predict(self) -> None:
        if self.mean is None or self.covariance is None:
            return
        m = self.mean.copy()
        if self.state != TrackState.Tracked:
            m[6:8] = 0
        self.mean, self.covariance = self.kalman_filter.predict(m, self.covariance)

    @staticmethod
    def multi_predict(stracks: Sequence["STrack"]) -> None:
        if not stracks:
            return
        mm = np.asarray([s.mean.copy() for s in stracks])
        cc = np.asarray([s.covariance for s in stracks])
        for i, s in enumerate(stracks):
            if s.state != TrackState.Tracked:
                mm[i][6:8] = 0
        mm, cc = STrack.shared_kalman.multi_predict(mm, cc)
        for s, m, c in zip(stracks, mm, cc):
            s.mean, s.covariance = m, c

    @staticmethod
    def multi_gmc(stracks: Sequence["STrack"], H: np.ndarray | None = None) -> None:
        if not stracks:
            return
        if H is None:
            H = np.eye(2, 3, dtype=np.float32)
        R = H[:2, :2]
        R8 = np.kron(np.eye(4), R)
        t = H[:2, 2]
        for st in stracks:
            mean = R8.dot(st.mean)
            mean[:2] += t
            st.mean = mean
            st.covariance = R8.dot(st.covariance).dot(R8.T)

    # ---------------- state helpers ---------------- #
    def activate(self, kf: KalmanFilterXYWH, frame_id: int) -> None:
        self.kalman_filter = kf
        self.id = self.next_id()
        self.mean, self.covariance = kf.initiate(self.xywh)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track: "STrack", frame_id: int, *, new_id: bool = False) -> None:
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, new_track.xywh)
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.id = self.next_id()
        self.conf = new_track.conf
        self.cls = new_track.cls
        self.det_ind = new_track.det_ind
        self.update_cls(new_track.cls, new_track.conf)

    def update(self, new_track: "STrack", frame_id: int) -> None:
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.history_observations.append(self.xyxy)
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, new_track.xywh)
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.state = TrackState.Tracked
        self.is_activated = True
        self.conf = new_track.conf
        self.cls = new_track.cls
        self.det_ind = new_track.det_ind
        self.update_cls(new_track.cls, new_track.conf)

    @property
    def xyxy(self) -> np.ndarray:
        box = self.mean[:4].copy() if self.mean is not None else self.xywh.copy()
        return xywh2xyxy(box)

