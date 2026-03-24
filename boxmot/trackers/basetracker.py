import colorsys
import hashlib
from abc import ABC, abstractmethod

import cv2 as cv
import numpy as np

from boxmot.trackers.detection_layout import (get_detection_layout,
                                              infer_detection_layout)
from boxmot.utils import logger as LOGGER
from boxmot.utils.iou import AssociationFunction
from boxmot.utils.visualization import VisualizationMixin


class BaseTracker(VisualizationMixin):
    supports_obb = False

    def __init__(
        self,
        det_thresh: float = 0.3,
        max_age: int = 30,
        max_obs: int = 50,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        per_class: bool = False,
        nr_classes: int = 80,
        asso_func: str = "iou",
        is_obb: bool = False,
        **kwargs,
    ):
        """
        Initialize the BaseTracker object

        Parameters:
        - det_thresh (float): Detection threshold for considering detections.
        - max_age (int): Maximum age (in frames) of a track before it is considered lost.
        - max_obs (int): Maximum number of historical observations (bounding boxes) stored for each track. max_obs is always greater than max_age by minimum 5.
        - min_hits (int): Minimum number of detection hits before a track is considered confirmed.
        - iou_threshold (float): IOU threshold for determining match between detection and tracks.
        - per_class (bool): Enables class-separated tracking
        - nr_classes (int): Total number of object classes that the tracker will handle (for per_class=True)
        - asso_func (str): Algorithm name used for data association between detections and tracks
            Options:
                - "iou" (default): Standard Intersection over Union
                - "iou_obb": IoU for oriented bounding boxes
                - "hmiou": Height-modified IoU that incorporates vertical overlap ratio
                - "giou": Generalized IoU that penalizes non-overlapping boxes
                - "ciou": Complete IoU with center point distance and aspect ratio consistency
                - "diou": Distance IoU that considers center point distance
                - "centroid": Distance between centroids of bounding boxes
                - "centroid_obb": Centroid distance for oriented bounding boxes
        - is_obb (bool): Work with Oriented Bounding Boxes (OBB) instead of standard axis-aligned bounding boxes?
                If False (default): If True: dets.shape[1] == 6, i.e. (x1,y1,x2,y2,conf,cls)
                If True: dets.shape[1] == 7, i.e. (cx,cy,w,h,angle,conf,cls)

        Attributes:
        - frame_count (int): Counter for the frames processed.
        - active_tracks (list): List to hold active tracks, may be used differently in subclasses.
        """

        self.det_thresh = det_thresh
        self.max_age = max_age
        self.max_obs = max_obs
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.per_class = per_class
        self.nr_classes = nr_classes
        self._asso_func_base_name = asso_func
        self.detection_layout = get_detection_layout(is_obb)
        self.asso_func_name = self.detection_layout.association_mode_name(asso_func)
        self.is_obb = self.detection_layout.is_obb

        # Attributes
        self.frame_count = 0
        self.active_tracks = []  # This might be handled differently in derived classes

        self.per_class_active_tracks = None
        self._first_frame_processed = (
            False  # Flag to track if the first frame has been processed
        )
        self._first_dets_processed = False
        self.last_emb_size = None  # Tracks the dimensionality of embedding vectors used for re-identification during tracking.

        # Initialize per-class active tracks
        if self.per_class:
            self.per_class_active_tracks = {}
            for i in range(self.nr_classes):
                self.per_class_active_tracks[i] = []

        if self.max_age >= self.max_obs:
            LOGGER.warning(
                "Max age > max observations, increasing size of max observations..."
            )
            self.max_obs = self.max_age + 5

        # Plotting lifecycle bookkeeping
        self._plot_frame_idx = -1
        self._removed_first_seen = {}
        self._removed_expired = set()
        self.removed_display_frames = getattr(self, "removed_display_frames", 10)

        # Log all params if tracker_name provided via kwargs
        tracker_name = kwargs.pop('_tracker_name', None)
        if tracker_name:
            base_params = {
                'det_thresh': det_thresh, 'max_age': max_age, 'max_obs': max_obs,
                'min_hits': min_hits, 'iou_threshold': iou_threshold, 'per_class': per_class,
                'asso_func': asso_func,
            }
            # Filter out internal/non-config params
            filtered_kwargs = {k: v for k, v in kwargs.items() 
                              if not k.startswith('_') and k not in ('__class__', 'reid_weights', 'device', 'half')}
            all_params = {**base_params, **filtered_kwargs}
            params_str = ", ".join(f"{k}={v}" for k, v in all_params.items())
            LOGGER.success(f"{tracker_name}: {params_str}")

    @abstractmethod
    def update(
        self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None
    ) -> np.ndarray:
        """
        Abstract method to update the tracker with new detections for a new frame. This method
        should be implemented by subclasses.

        Parameters:
        - dets (np.ndarray): Array of detections for the current frame.
        - img (np.ndarray): The current frame as an image array.
        - embs (np.ndarray, optional): Embeddings associated with the detections, if any.

        Raises:
        - NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError(
            "The update method needs to be implemented by the subclass."
        )

    def get_class_dets_n_embs(self, dets, embs, cls_id):
        # Initialize empty arrays for detections and embeddings
        class_dets = self.detection_layout.empty_dets(dtype=np.float32)
        class_embs = (
            np.empty((0, self.last_emb_size))
            if self.last_emb_size is not None
            else None
        )

        # Check if there are detections
        if dets.size == 0:
            return class_dets, class_embs

        class_indices = np.where(dets[:, self.detection_layout.cls_idx] == cls_id)[0]
        class_dets = dets[class_indices]

        if embs is None:
            return class_dets, class_embs

        # Assert that if embeddings are provided, they have the same number of elements as detections
        assert dets.shape[0] == embs.shape[0], (
            "Detections and embeddings must have the same number of elements when both are provided"
        )
        class_embs = None
        if embs.size > 0:
            class_embs = embs[class_indices]
            self.last_emb_size = class_embs.shape[
                1
            ]  # Update the last known embedding size
        return class_dets, class_embs

    def _set_detection_mode(self, is_obb: bool) -> None:
        """Update the tracker detection mode and association function name."""
        self.detection_layout = get_detection_layout(is_obb)
        self.is_obb = self.detection_layout.is_obb
        self.asso_func_name = self.detection_layout.association_mode_name(
            self._asso_func_base_name
        )

        if self._first_frame_processed and hasattr(self, "w") and hasattr(self, "h"):
            self.asso_func = AssociationFunction(
                w=self.w, h=self.h, asso_mode=self.asso_func_name
            ).asso_func

    def empty_detections(self, dtype=np.float32) -> np.ndarray:
        return self.detection_layout.empty_dets(dtype=dtype)

    def empty_output(self, dtype=float) -> np.ndarray:
        return self.detection_layout.empty_output(dtype=dtype)

    @staticmethod
    def setup_decorator(method):
        """
        Decorator to perform setup on the first frame only.
        This ensures that initialization tasks (like setting the association function) only
        happen once, on the first frame, and are skipped on subsequent frames.
        """

        def wrapper(self, *args, **kwargs):
            # Extract detections and image from args
            dets = args[0]
            img = args[1] if len(args) > 1 else None

            # Unwrap `data` attribute if present
            if hasattr(dets, "data"):
                dets = dets.data

            # Convert memoryview to numpy array if needed
            if isinstance(dets, memoryview):
                dets = np.array(dets, dtype=np.float32)  # Adjust dtype if needed

            # First-time detection setup
            if not self._first_dets_processed and dets is not None:
                layout = infer_detection_layout(dets)
                if layout is not None:
                    if layout.is_obb and not self.supports_obb:
                        raise AssertionError(
                            f"{self.__class__.__name__} does not support OBB detections. "
                            "Use an OBB-capable tracker such as ByteTrack, BotSort, OCSort, or SFSORT."
                        )
                    self._set_detection_mode(layout.is_obb)
                    self._first_dets_processed = True

            # First frame image-based setup
            if not self._first_frame_processed and img is not None:
                self.h, self.w = img.shape[0:2]
                self.asso_func = AssociationFunction(
                    w=self.w, h=self.h, asso_mode=self.asso_func_name
                ).asso_func
                self._first_frame_processed = True

            # Call the original method with the unwrapped `dets`
            return method(self, dets, img, *args[2:], **kwargs)

        return wrapper

    @staticmethod
    def per_class_decorator(update_method):
        """
        Decorator for the update method to handle per-class processing.
        """

        def wrapper(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None):
            # handle different types of inputs
            if dets is None or len(dets) == 0:
                dets = self.empty_detections()

            if not self.per_class:
                # Process all detections at once if per_class is False
                return update_method(self, dets=dets, img=img, embs=embs)
            # else:
            # Initialize an array to store the tracks for each class
            per_class_tracks = []

            # same frame count for all classes
            frame_count = self.frame_count

            for cls_id in range(self.nr_classes):
                # Get detections and embeddings for the current class
                class_dets, class_embs = self.get_class_dets_n_embs(dets, embs, cls_id)

                LOGGER.debug(
                    f"Processing class {int(cls_id)}: {class_dets.shape} with embeddings"
                    f" {class_embs.shape if class_embs is not None else None}"
                )

                # Activate the specific active tracks for this class id
                self.active_tracks = self.per_class_active_tracks[cls_id]

                # Reset frame count for every class
                self.frame_count = frame_count

                # Update detections using the decorated method
                tracks = update_method(self, dets=class_dets, img=img, embs=class_embs)

                # Save the updated active tracks
                self.per_class_active_tracks[cls_id] = self.active_tracks

                if tracks.size > 0:
                    per_class_tracks.append(tracks)

            # Increase frame count by 1
            self.frame_count = frame_count + 1
            if per_class_tracks:
                return np.vstack(per_class_tracks)

            return self.empty_output()

        return wrapper

    def check_inputs(self, dets, img, embs=None):
        assert isinstance(dets, np.ndarray), (
            f"Unsupported 'dets' input format '{type(dets)}', valid format is np.ndarray"
        )
        assert isinstance(img, np.ndarray), (
            f"Unsupported 'img_numpy' input format '{type(img)}', valid format is np.ndarray"
        )
        assert len(dets.shape) == 2, (
            "Unsupported 'dets' dimensions, valid number of dimensions is two"
        )

        if embs is not None:
            assert dets.shape[0] == embs.shape[0], (
                "Missmatch between detections and embeddings sizes"
            )

        self.detection_layout.validate_dets(dets)

    def get_active_tracks_for_display(self) -> list:
        """Return the currently active tracks, flattened across classes if needed."""
        if self.per_class_active_tracks is None:
            return list(self.active_tracks or [])

        tracks = []
        for class_tracks in self.per_class_active_tracks.values():
            tracks.extend(class_tracks)
        return tracks

    def get_lost_tracks_for_display(self) -> list:
        """Return lost tracks when the tracker maintains an explicit lost list."""
        return list(getattr(self, "lost_stracks", []) or [])

    def get_removed_tracks_for_display(self) -> list:
        """Return removed tracks when the tracker maintains an explicit removed list."""
        return list(getattr(self, "removed_stracks", []) or [])

    def get_track_history_for_display(self, track) -> list:
        """Return the stored observation history used to draw trajectories."""
        return list(getattr(track, "history_observations", []) or [])

    def get_track_state_for_display(self, track):
        """Infer a generic lifecycle state for trackers without explicit state lists."""
        if hasattr(track, "hits") and track.hits < self.min_hits:
            return None
        if hasattr(track, "is_activated") and not track.is_activated:
            return None

        if hasattr(track, "time_since_update"):
            if track.time_since_update == 0:
                return "confirmed"
            if track.time_since_update <= self.max_age:
                return "predicted"
            return "lost"

        if hasattr(track, "state"):
            try:
                from boxmot.trackers.bytetrack.basetrack import TrackState

                if track.state == TrackState.Tracked:
                    return "confirmed"
                if track.state == TrackState.Lost:
                    return "predicted"
                return "lost"
            except Exception:
                return "confirmed" if getattr(track, "is_activated", True) else "lost"

        return "confirmed"

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
        return (getattr(self, "lost_stracks", None) is not None) or (
            getattr(self, "removed_stracks", None) is not None
        )

    def _removed_track_display_key(self, track):
        start_frame = int(getattr(track, "start_frame", getattr(track, "birth_frame", -1)))
        track_id = self.get_track_id_for_display(track)
        return (track_id, start_frame) if start_frame >= 0 else track_id

    def _get_removed_tracks_for_display(self, now: int, ttl: int) -> list:
        """Return removed tracks that should remain visible for the current plot frame."""
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
        """Trim old removed-track tombstones so lifecycle bookkeeping stays bounded."""
        if len(self._removed_expired) <= 10000:
            return

        horizon = getattr(self, "removed_tombstone_horizon", 10000)
        cutoff = now - max(ttl, 1) - horizon
        stale_keys = [
            key
            for key, first_seen in self._removed_first_seen.items()
            if first_seen < cutoff
        ]
        for key in stale_keys:
            self._removed_first_seen.pop(key, None)
            self._removed_expired.discard(key)

    def _display_groups_with_explicit_lifecycle(self, active_tracks: list):
        """Yield display groups for trackers with explicit active/lost/removed lists."""
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
        """Yield track groups for visualization as (tracks, forced_state, style)."""
        self._plot_frame_idx += 1

        active_tracks = self.get_active_tracks_for_display()
        if self.has_explicit_display_lifecycle():
            yield from self._display_groups_with_explicit_lifecycle(active_tracks)
            return

        if active_tracks:
            yield (active_tracks, None, "dashed")

    def iter_tracks_for_display(self, show_kf_preds: bool = False):
        """Yield individual tracks as (track, state, style) for rendering."""
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

    def reset(self):
        pass
