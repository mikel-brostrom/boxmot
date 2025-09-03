import colorsys
import hashlib
from abc import ABC, abstractmethod

import cv2 as cv
import numpy as np

from boxmot.utils import logger as LOGGER
from boxmot.utils.iou import AssociationFunction


class BaseTracker(ABC):
    def __init__(
        self,
        det_thresh: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        max_obs: int = 50,
        nr_classes: int = 80,
        per_class: bool = False,
        asso_func: str = "iou",
        is_obb: bool = False,
    ):
        self.det_thresh = det_thresh
        self.max_age = max_age
        self.max_obs = max_obs
        self.min_hits = min_hits
        self.per_class = per_class
        self.nr_classes = nr_classes
        self.iou_threshold = iou_threshold
        self.last_emb_size = None
        self.asso_func_name = asso_func + "_obb" if is_obb else asso_func
        self.is_obb = is_obb

        self.frame_count = 0
        self.active_tracks = []
        self.per_class_active_tracks = None
        self._first_frame_processed = False
        self._first_dets_processed = False

        if self.per_class:
            self.per_class_active_tracks = {i: [] for i in range(self.nr_classes)}

        if self.max_age >= self.max_obs:
            LOGGER.warning("Max age > max observations, increasing size of max observations...")
            self.max_obs = self.max_age + 5

        # Plotting lifecycle bookkeeping
        self._plot_frame_idx = -1
        self._removed_first_seen = {}
        self._removed_expired = set()
        self.removed_display_frames = getattr(self, "removed_display_frames", 10)

    @abstractmethod
    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None) -> np.ndarray:
        raise NotImplementedError("The update method needs to be implemented by the subclass.")

    def get_class_dets_n_embs(self, dets, embs, cls_id):
        class_dets = np.empty((0, 6))
        class_embs = np.empty((0, self.last_emb_size)) if self.last_emb_size is not None else None

        if dets.size == 0:
            return class_dets, class_embs

        class_indices = np.where(dets[:, 5] == cls_id)[0]
        class_dets = dets[class_indices]

        if embs is None:
            return class_dets, class_embs

        assert dets.shape[0] == embs.shape[0], (
            "Detections and embeddings must have the same number of elements when both are provided"
        )
        class_embs = None
        if embs.size > 0:
            class_embs = embs[class_indices]
            self.last_emb_size = class_embs.shape[1]
        return class_dets, class_embs

    @staticmethod
    def setup_decorator(method):
        def wrapper(self, *args, **kwargs):
            dets = args[0]
            img = args[1] if len(args) > 1 else None

            if hasattr(dets, 'data'):
                dets = dets.data
            if isinstance(dets, memoryview):
                dets = np.array(dets, dtype=np.float32)

            if not self._first_dets_processed and dets is not None:
                if dets.ndim == 2 and dets.shape[1] == 6:
                    self.is_obb = False
                    self._first_dets_processed = True
                elif dets.ndim == 2 and dets.shape[1] == 7:
                    self.is_obb = True
                    self._first_dets_processed = True

            if not self._first_frame_processed and img is not None:
                self.h, self.w = img.shape[0:2]
                self.asso_func = AssociationFunction(
                    w=self.w,
                    h=self.h,
                    asso_mode=self.asso_func_name
                ).asso_func
                self._first_frame_processed = True

            return method(self, dets, img, *args[2:], **kwargs)
        return wrapper

    @staticmethod
    def per_class_decorator(update_method):
        def wrapper(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None):
            if dets is None or len(dets) == 0:
                dets = np.empty((0, 6))

            if not self.per_class:
                return update_method(self, dets=dets, img=img, embs=embs)

            per_class_tracks = []
            frame_count = self.frame_count

            for cls_id in range(self.nr_classes):
                class_dets, class_embs = self.get_class_dets_n_embs(dets, embs, cls_id)
                LOGGER.debug(
                    f"Processing class {int(cls_id)}: {class_dets.shape} with embeddings "
                    f"{class_embs.shape if class_embs is not None else None}"
                )

                self.active_tracks = self.per_class_active_tracks[cls_id]
                self.frame_count = frame_count

                tracks = update_method(self, dets=class_dets, img=img, embs=class_embs)
                self.per_class_active_tracks[cls_id] = self.active_tracks

                if tracks.size > 0:
                    per_class_tracks.append(tracks)

            self.frame_count = frame_count + 1
            return np.vstack(per_class_tracks) if per_class_tracks else np.empty((0, 8))
        return wrapper

    def check_inputs(self, dets, img, embs=None):
        assert isinstance(dets, np.ndarray), f"Unsupported 'dets' input format '{type(dets)}', valid format is np.ndarray"
        assert isinstance(img, np.ndarray), f"Unsupported 'img_numpy' input format '{type(img)}', valid format is np.ndarray"
        assert len(dets.shape) == 2, "Unsupported 'dets' dimensions, valid number of dimensions is two"

        if embs is not None:
            assert dets.shape[0] == embs.shape[0], "Missmatch between detections and embeddings sizes"

        if self.is_obb:
            assert dets.shape[1] == 7, "Unsupported 'dets' 2nd dimension length, valid length is 7 (cx,cy,w,h,angle,conf,cls)"
        else:
            assert dets.shape[1] == 6, "Unsupported 'dets' 2nd dimension length, valid lengths is 6 (x1,y1,x2,y2,conf,cls)"

    def id_to_color(
        self,
        id: int,
        saturation: float = 0.75,
        value: float = 0.95,
        state: str = "confirmed"
    ) -> tuple:
        target_id = getattr(self, "target_id", None)
        if target_id is not None:
            return (0, 255, 0) if id == target_id else (0, 0, 0)

        if state == "confirmed":
            return (0, 255, 0)    # Green
        elif state == "predicted":
            return (0, 165, 255)  # Orange
        elif state == "lost":
            return (0, 0, 255)    # Red
        elif state == "removed":
            return (0, 0, 255)    # Red

        # Default: consistent hashed color for ID
        hash_object = hashlib.sha256(str(id).encode())
        hue = int(hash_object.hexdigest()[:8], 16) / 0xFFFFFFFF
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        rgb_255 = tuple(int(component * 255) for component in rgb)
        return rgb_255[::-1]

    def _draw_dashed_rect(self, img, x1, y1, x2, y2, color, thickness, dash=4, gap=2):
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
        color = self.id_to_color(id, state=state)
        if self.is_obb:
            # OBB: draw solid poly; dashed fallback to solid
            angle = box[4] * 180.0 / np.pi  # radians -> degrees
            box_poly = ((box[0], box[1]), (box[2], box[3]), angle)
            rotrec = cv.boxPoints(box_poly)
            box_poly = np.int_(rotrec)
            img = cv.polylines(img, [box_poly], isClosed=True, color=color, thickness=thickness)
            label = f"id: {int(id)}, conf: {conf:.2f}, c: {int(cls)}, a: {box[4]:.2f}"
            img = cv.putText(img, label, (int(box[0]), int(box[1]) - 10),
                             cv.FONT_HERSHEY_SIMPLEX, fontscale, color, thickness)
        else:
            x1, y1, x2, y2 = map(int, (box[0], box[1], box[2], box[3]))
            if style == "dashed":
                img = self._draw_dashed_rect(img, x1, y1, x2, y2, color, thickness)
            else:
                img = cv.rectangle(img, (x1, y1), (x2, y2), color, thickness)
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

    def _display_groups(self):
        """
        Yield groups of (tracks, forced_state, style) ready for drawing.
        If ByteTrack-style lists exist, use them with styles and TTLs.
        Otherwise, fall back to all active tracks and per-track inferred state.
        """
        lost_list = getattr(self, "lost_stracks", None)
        removed_list = getattr(self, "removed_stracks", None)

        # Maintain internal frame index for TTL accounting
        self._plot_frame_idx += 1
        now = self._plot_frame_idx

        ttl = int(max(0, getattr(self, "removed_display_frames", self.removed_display_frames)))

        if (lost_list is not None) or (removed_list is not None):
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

        else:
            # Generic fallback: only active tracks; state per track
            active_tracks = self._all_active_tracks()
            if active_tracks:
                yield (active_tracks, None, "solid")

    def _draw_track(self, img, a, forced_state, style, thickness, fontscale, show_trajectories):
        if not getattr(a, "history_observations", None):
            return img

        state = forced_state or self._infer_state(a)
        if state is None:
            return img  # e.g., below min_hits

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
    ) -> np.ndarray:
        """
        DRY visualization for active/lost/removed tracks with optional trajectories.
        - Uses ByteTrack-style lists if available (lost/removed), including TTL for removed.
        - Otherwise falls back to inferring state from each active track.
        - OBBs are always drawn solid (dashed is AABB-only).
        """
        for tracks, forced_state, style in self._display_groups():
            for a in tracks:
                img = self._draw_track(
                    img, a, forced_state=forced_state, style=style,
                    thickness=thickness, fontscale=fontscale, show_trajectories=show_trajectories
                )
        return img

    def reset(self):
        pass
