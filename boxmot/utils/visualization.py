import colorsys
import hashlib
from abc import ABC, abstractmethod

import cv2 as cv
import numpy as np


class BaseVisualization(ABC):
    """
    Abstract base class for visualization methods in BaseTracker.
    """
    
    def id_to_color(
        self,
        id: int,
        saturation: float = 0.75,
        value: float = 0.95,
        state: str = "confirmed"
    ) -> tuple:
        """
        Returns green for target_id, otherwise generates a consistent unique BGR color using ID hashing.
        """
        target_id = getattr(self, "target_id", None)
        if target_id is not None:
            return (0, 255, 0) if id == target_id else (0, 0, 0)

        # Default: consistent hashed color for other IDs
        hash_object = hashlib.sha256(str(id).encode())
        hash_digest = hash_object.hexdigest()
        hue = int(hash_digest[:8], 16) / 0xFFFFFFFF

        # Convert HSV to RGB
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        rgb_255 = tuple(int(component * 255) for component in rgb)

        # Convert to BGR
        return rgb_255[::-1]

    def _draw_dashed_rect(self, img, x1, y1, x2, y2, color, thickness, dash=10, gap=10):
        """Dashed rectangle for AABB (used for 'lost'/'predicted')."""
        # Top / Bottom
        for i in range(x1, x2, dash + gap):
            img = cv.line(img, (i, y1), (min(i + dash, x2), y1), color, thickness)
            img = cv.line(img, (i, y2), (min(i + dash, x2), y2), color, thickness)
        # Left / Right
        for i in range(y1, y2, dash + gap):
            img = cv.line(img, (x1, i), (x1, min(i + dash, y2)), color, thickness)
            img = cv.line(img, (x2, i), (x2, min(i + dash, y2)), color, thickness)
        return img

    def plot_box_on_img(
        self,
        img: np.ndarray,
        box: tuple,
        conf: float,
        cls: int,
        id: int,
        thickness: int = 2,
        fontscale: float = 0.5,
        state: str = "confirmed",
        style: str = "solid",  # "solid" | "dashed" (dashed only for AABB)
    ) -> np.ndarray:
        """
        Draws a bounding box with ID, confidence, and class information on an image.
        """
        color = self.id_to_color(id, state=state)
        if self.is_obb:
            angle = box[4] * 180.0 / np.pi  # Convert radians to degrees
            box_poly = ((box[0], box[1]), (box[2], box[3]), angle)
            rotrec = cv.boxPoints(box_poly)
            box_poly = np.int_(rotrec)  # Convert to integer

            # Draw the rectangle on the image
            img = cv.polylines(
                img,
                [box_poly],
                isClosed=True,
                color=color,
                thickness=thickness,
            )

            img = cv.putText(
                img,
                f"id: {int(id)}, conf: {conf:.2f}, c: {int(cls)}, a: {box[4]:.2f}",
                (int(box[0]), int(box[1]) - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                fontscale,
                color,
                thickness,
            )
        else:
            x1, y1, x2, y2 = map(int, (box[0], box[1], box[2], box[3]))
            if style == "dashed":
                img = self._draw_dashed_rect(img, x1, y1, x2, y2, color, thickness)
            else:
                img = cv.rectangle(
                    img,
                    (x1, y1),
                    (x2, y2),
                    color,
                    thickness,
                )
            img = cv.putText(
                img,
                f"id: {int(id)}, conf: {conf:.2f}, c: {int(cls)}",
                (x1, max(0, y1 - 10)),
                cv.FONT_HERSHEY_SIMPLEX,
                fontscale,
                color,
                thickness,
            )
        return img

    def plot_trackers_trajectories(
        self, img: np.ndarray, observations: list, id: int, state: str = "confirmed"
    ) -> np.ndarray:
        """
        Draws the trajectories of tracked objects based on historical observations.
        """
        for i, box in enumerate(observations):
            trajectory_thickness = int(np.sqrt(float(i + 1)) * 1.2)
            if self.is_obb:
                img = cv.circle(
                    img,
                    (int(box[0]), int(box[1])),
                    2,
                    color=self.id_to_color(int(id), state=state),
                    thickness=trajectory_thickness,
                )
            else:
                img = cv.circle(
                    img,
                    (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)),
                    2,
                    color=self.id_to_color(int(id), state=state),
                    thickness=trajectory_thickness,
                )
        return img

    # ---------- Helpers for a DRY plot_results ----------
    def _all_active_tracks(self):
        """Flatten active tracks across classes (if per-class)."""
        if getattr(self, "per_class_active_tracks", None) is None:
            return list(getattr(self, "active_tracks", []) or [])
        tracks = []
        for k in self.per_class_active_tracks.keys():
            tracks += self.per_class_active_tracks[k]
        return tracks

    def _infer_state(self, a):
        """Infer a generic state string for a track when lost/removed lists are not present."""
        if hasattr(a, "hits"):  # DeepOCSort / OCSort
            if a.hits < getattr(self, "min_hits", 0):
                return None  # not yet confirmed -> skip
        elif hasattr(a, "is_activated"):  # ByteTrack
            if not a.is_activated:
                return None

        if hasattr(a, "time_since_update"):
            if a.time_since_update == 0:
                return "confirmed"
            elif a.time_since_update <= getattr(self, "max_age", 1_000_000):
                return "predicted"
            else:
                return "lost"

        if hasattr(a, "state"):
            try:
                from boxmot.trackers.bytetrack.basetrack import TrackState
                if a.state == TrackState.Tracked:
                    return "confirmed"
                elif a.state == TrackState.Lost:
                    return "predicted"
                else:
                    return "lost"
            except Exception:
                return "confirmed" if getattr(a, "is_activated", True) else "lost"

        return "confirmed"

    @abstractmethod
    def _display_groups(self):
        pass

    def _draw_track(self, img, a, forced_state, style, thickness, fontscale, show_trajectories):
        if not getattr(a, "history_observations", None):
            return img

        state = forced_state or self._infer_state(a)
        if state is None:
            return img  # e.g., below min_hits

        # If the track is lost (predicted), use the current Kalman Filter prediction
        # instead of the last history observation (which is the last seen detection)
        if state == "predicted":
            if hasattr(a, "xyxy"):
                box = a.xyxy
            elif hasattr(a, "get_state"):
                box = a.get_state()
                # Handle OCSORT's get_state returning (1, 4) array
                if isinstance(box, np.ndarray) and box.ndim == 2 and box.shape[0] == 1:
                    box = box[0]
            else:
                box = a.history_observations[-1]
        else:
            box = a.history_observations[-1]

        conf = getattr(a, "conf", 1.0)
        cls = getattr(a, "cls", -1)

        img = self.plot_box_on_img(
            img=img,
            box=box,
            conf=conf,
            cls=cls,
            id=int(getattr(a, "id")),
            thickness=thickness,
            fontscale=fontscale,
            state=state,
            style=style if (state == "predicted" and not self.is_obb) else "solid",
        )

        if show_trajectories:
            img = self.plot_trackers_trajectories(img, a.history_observations, int(getattr(a, "id")), state=state)
        return img

    def plot_results(
        self,
        img: np.ndarray,
        show_trajectories: bool,
        thickness: int = 2,
        fontscale: float = 0.5,
        show_lost: bool = False,
    ) -> np.ndarray:
        """
        Visualizes the trajectories of all active tracks on the image.
        """
        for tracks, forced_state, style in self._display_groups():
            if not show_lost and forced_state in ("predicted", "removed"):
                continue

            for a in tracks:
                if not show_lost and forced_state is None:
                    state = self._infer_state(a)
                    if state != "confirmed":
                        continue

                img = self._draw_track(
                    img,
                    a,
                    forced_state=forced_state,
                    style=style,
                    thickness=thickness,
                    fontscale=fontscale,
                    show_trajectories=show_trajectories,
                )
        return img


class ExplicitStateVisualization(BaseVisualization):
    """
    Visualization for trackers that maintain explicit lists for lost and removed tracks.
    """

    def _display_groups(self):
        lost_list = getattr(self, "lost_stracks", None)
        removed_list = getattr(self, "removed_stracks", None)

        # Maintain internal frame index for TTL accounting
        self._plot_frame_idx += 1
        now = self._plot_frame_idx

        ttl = int(max(0, getattr(self, "removed_display_frames", self.removed_display_frames)))

        # Active
        yield (self._all_active_tracks(), "confirmed", "solid")

        # Lost (dashed, orange)
        if lost_list:
            yield (list(lost_list), "predicted", "dashed")

        # Removed (gray, solid), with TTL + tombstone
        if removed_list and ttl > 0:
            filtered_removed = []
            for a in removed_list:
                if not getattr(a, "history_observations", None):
                    continue
                sf = int(getattr(a, "start_frame", getattr(a, "birth_frame", -1)))
                rid = int(getattr(a, "id"))
                key = (rid, sf) if sf >= 0 else rid

                if key in self._removed_expired:
                    continue

                if key not in self._removed_first_seen:
                    self._removed_first_seen[key] = now

                if (now - self._removed_first_seen[key]) < ttl:
                    filtered_removed.append(a)
                else:
                    self._removed_expired.add(key)

            if filtered_removed:
                yield (filtered_removed, "removed", "solid")

        # Optional: simple memory cap
        if len(self._removed_expired) > 10000:
            horizon = getattr(self, "removed_tombstone_horizon", 10000)
            cutoff = now - max(ttl, 1) - horizon
            to_drop = [k for k, t0 in self._removed_first_seen.items() if t0 < cutoff]
            for k in to_drop:
                self._removed_first_seen.pop(k, None)
                self._removed_expired.discard(k)


class InferredStateVisualization(BaseVisualization):
    """
    Visualization for trackers that only expose active tracks and state is inferred.
    """

    def _display_groups(self):
        # Maintain internal frame index for TTL accounting
        self._plot_frame_idx += 1
        
        # Generic fallback: only active tracks; state per track
        active_tracks = self._all_active_tracks()
        if active_tracks:
            yield (active_tracks, None, "dashed")


class VisualizationMixin(BaseVisualization):
    """
    Mixin class for visualization methods in BaseTracker.
    """
    
    def _display_groups(self):
        lost_list = getattr(self, "lost_stracks", None)
        removed_list = getattr(self, "removed_stracks", None)
        
        if (lost_list is not None) or (removed_list is not None):
            return ExplicitStateVisualization._display_groups(self)
        else:
            return InferredStateVisualization._display_groups(self)
