from __future__ import annotations

import numpy as np

from boxmot.trackers.common.tracking.collections import (
    owner_has_track_collection,
    tracks_from_owner,
)
from boxmot.trackers.common.tracking.track import TrackState


class TrackDisplayMixin:
    """Lifecycle and geometry adapters used by tracker visualization."""

    def get_active_tracks_for_display(self) -> list:
        """Return tracks that should be drawn as active."""
        if self.class_track_states is not None:
            tracks = self.all_class_tracks("active")
            return tracks if tracks else self.all_class_tracks("pool")
        return list(self.active_tracks or [])

    def get_lost_tracks_for_display(self) -> list:
        """Return tracks that should be drawn as predicted/lost."""
        if self.class_track_states is not None:
            return self.all_class_tracks("lost")
        return tracks_from_owner(self, "lost")

    def get_removed_tracks_for_display(self) -> list:
        """Return removed tracks that should stay visible temporarily."""
        if self.class_track_states is not None:
            return self.all_class_tracks("removed")
        return tracks_from_owner(self, "removed")

    def get_track_history_for_display(self, track) -> list:
        """Return stored observations used to draw trajectories."""
        return list(getattr(track, "history_observations", []) or [])

    def get_track_state_for_display(self, track):
        """Infer a display lifecycle state for a tracker-local track object."""
        if hasattr(track, "hits") and track.hits < self.min_hits:
            return None
        if hasattr(track, "is_activated") and not track.is_activated:
            return None

        meta_state = getattr(getattr(track, "meta", None), "state", None)
        display_state = self._display_state_from_common_state(meta_state)
        if display_state in ("predicted", "removed"):
            return display_state

        if hasattr(track, "time_since_update"):
            if track.time_since_update == 0:
                return "confirmed"
            if track.time_since_update <= self.max_age:
                return "predicted"
            return "lost"

        if display_state is not None:
            return display_state

        if hasattr(track, "state"):
            return self._display_state_from_local_state(track)

        return "confirmed"

    @staticmethod
    def _display_state_from_common_state(state) -> str | None:
        if state is TrackState.TRACKED:
            return "confirmed"
        if state is TrackState.LOST:
            return "predicted"
        if state is TrackState.REMOVED:
            return "removed"
        return None

    @staticmethod
    def _display_state_from_local_state(track) -> str:
        state = getattr(track, "state", None)
        state_name = getattr(state, "name", state if isinstance(state, str) else None)
        if state_name is not None:
            normalized = str(state_name).replace("_", "").replace("-", "").lower()
            if normalized in {"tracked", "confirmed", "active", "reliable"}:
                return "confirmed"
            if normalized in {"lost", "longlost", "lostcentral", "lostmarginal", "suspicious"}:
                return "predicted"
            if normalized in {"removed", "deleted", "frameout"}:
                return "removed"
            if normalized in {"new", "tentative", "pending"}:
                return "lost"

        module_name = getattr(track.__class__, "__module__", "")
        byte_or_bot_state = module_name in {
            "boxmot.trackers.bbox.bytetrack",
            "boxmot.trackers.bbox.botsort",
        }
        if isinstance(state, (int, np.integer)) and byte_or_bot_state:
            if int(state) == 1:
                return "confirmed"
            if int(state) == 2:
                return "predicted"
            return "lost"

        return "confirmed" if getattr(track, "is_activated", True) else "lost"

    def get_track_id_for_display(self, track) -> int:
        return int(getattr(track, "id"))

    def get_track_conf_for_display(self, track) -> float:
        return float(getattr(track, "conf", 1.0))

    def get_track_cls_for_display(self, track) -> int:
        return int(getattr(track, "cls", -1))

    @staticmethod
    def _resolve_track_box_attr(track, attr_name):
        if not hasattr(track, attr_name):
            return None

        value = getattr(track, attr_name)
        if callable(value):
            value = value()
        if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[0] == 1:
            return value[0]
        return value

    def get_track_box_for_display(self, track, state: str):
        """Return the geometry that should be drawn for a given track state."""
        history = self.get_track_history_for_display(track)
        if state not in ("predicted", "removed"):
            return history[-1] if history else None

        if self.is_obb:
            for attr_name in ("_state_obb_for_plot", "xywha", "get_state", "xyxy"):
                box = self._resolve_track_box_attr(track, attr_name)
                if box is not None:
                    return box
        else:
            for attr_name in ("xyxy", "get_state"):
                box = self._resolve_track_box_attr(track, attr_name)
                if box is not None:
                    return box

        return history[-1] if history else None

    def has_explicit_display_lifecycle(self) -> bool:
        return owner_has_track_collection(self, "lost") or owner_has_track_collection(self, "removed")

    def _removed_track_display_key(self, track):
        start_frame = int(getattr(track, "start_frame", getattr(track, "birth_frame", -1)))
        track_id = self.get_track_id_for_display(track)
        return (track_id, start_frame) if start_frame >= 0 else track_id

    def _get_removed_tracks_for_display(self, now: int, ttl: int) -> list:
        """Return removed tracks that should remain visible for this frame."""
        if ttl <= 0:
            return []

        visible_tracks = []
        for track in self.get_removed_tracks_for_display():
            if not self.get_track_history_for_display(track):
                continue

            key = self._removed_track_display_key(track)
            if key in self._removed_expired:
                continue

            first_seen = self._removed_first_seen.setdefault(key, now)
            if (now - first_seen) < ttl:
                visible_tracks.append(track)
            else:
                self._removed_expired.add(key)

        return visible_tracks

    def _prune_removed_display_tombstones(self, now: int, ttl: int) -> None:
        """Trim old removed-track display keys so bookkeeping stays bounded."""
        if len(self._removed_expired) <= 10000:
            return

        horizon = getattr(self, "removed_tombstone_horizon", 10000)
        cutoff = now - max(ttl, 1) - horizon
        stale_keys = [key for key, first_seen in self._removed_first_seen.items() if first_seen < cutoff]
        for key in stale_keys:
            self._removed_first_seen.pop(key, None)
            self._removed_expired.discard(key)

    def _display_groups_with_explicit_lifecycle(self, active_tracks: list):
        """Yield display groups for trackers with explicit lifecycle lists."""
        now = self._plot_frame_idx
        ttl = int(max(0, self.removed_display_frames))

        yield (active_tracks, "confirmed", "solid")

        lost_tracks = self.get_lost_tracks_for_display()
        if lost_tracks:
            yield (lost_tracks, "predicted", "dashed")

        removed_tracks = self._get_removed_tracks_for_display(now=now, ttl=ttl)
        if removed_tracks:
            yield (removed_tracks, "removed", "solid")

        self._prune_removed_display_tombstones(now=now, ttl=ttl)

    def _display_groups(self):
        """Yield track groups as ``(tracks, forced_state, style)``."""
        self._plot_frame_idx += 1

        active_tracks = self.get_active_tracks_for_display()
        if self.has_explicit_display_lifecycle():
            yield from self._display_groups_with_explicit_lifecycle(active_tracks)
            return

        if active_tracks:
            yield (active_tracks, None, "dashed")

    def iter_tracks_for_display(self, show_kf_preds: bool = False):
        """Yield individual tracks as ``(track, state, style)``."""
        for tracks, forced_state, style in self._display_groups():
            if not show_kf_preds and forced_state in ("predicted", "removed"):
                continue

            for track in tracks:
                state = forced_state or self.get_track_state_for_display(track)
                if state is None:
                    continue
                if not show_kf_preds and state != "confirmed":
                    continue
                yield track, state, style
