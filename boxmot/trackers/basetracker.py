from abc import abstractmethod

import numpy as np

from boxmot.trackers.common.association.iou import AssociationFunction
from boxmot.trackers.common.detections import DetectionBatch
from boxmot.trackers.common.detections.layout import get_detection_layout, infer_detection_layout
from boxmot.trackers.common.motion import cmc as cmc_utils
from boxmot.trackers.common.tracking import outputs as output_utils
from boxmot.trackers.common.tracking.records import DetectionRecord, TrackRecord
from boxmot.trackers.common.tracking.track import TrackIdAllocator, TrackState
from boxmot.trackers.common.tracking.visualization import VisualizationMixin
from boxmot.trackers.track_results import TrackResults
from boxmot.utils import logger as LOGGER


class BaseTracker(VisualizationMixin):
    supports_obb = False
    supports_masks = False

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
        - max_obs (int): Maximum number of historical observations stored for each track.
          max_obs is always greater than max_age by minimum 5.
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

        tracker_name = kwargs.pop("_tracker_name", None)
        self.name = str(tracker_name or self.__class__.__name__)
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
        self.id_allocator = TrackIdAllocator()

        # Attributes
        self.frame_count = 0
        self.active_tracks = []  # This might be handled differently in derived classes

        self.per_class_active_tracks = None
        self._first_frame_processed = False  # Flag to track if the first frame has been processed
        self._first_dets_processed = False
        self.last_emb_size = None

        # Initialize per-class active tracks
        if self.per_class:
            self.per_class_active_tracks = {}
            self.per_class_trackers = {}
            for i in range(self.nr_classes):
                self.per_class_active_tracks[i] = []
                self.per_class_trackers[i] = []

        if self.max_age >= self.max_obs:
            LOGGER.warning("Max age > max observations, increasing size of max observations...")
            self.max_obs = self.max_age + 5

        # Plotting lifecycle bookkeeping
        self._plot_frame_idx = -1
        self._removed_first_seen = {}
        self._removed_expired = set()
        self.removed_display_frames = getattr(self, "removed_display_frames", 10)

        # Log all params if tracker_name provided via kwargs
        if tracker_name:
            base_params = {
                "det_thresh": det_thresh,
                "max_age": max_age,
                "max_obs": max_obs,
                "min_hits": min_hits,
                "iou_threshold": iou_threshold,
                "per_class": per_class,
                "asso_func": asso_func,
            }
            # Filter out internal/non-config params
            filtered_kwargs = {
                k: v
                for k, v in kwargs.items()
                if not k.startswith("_") and k not in ("__class__", "reid_weights", "device", "half")
            }
            all_params = {**base_params, **filtered_kwargs}
            params_str = ", ".join(f"{k}={v}" for k, v in all_params.items())
            LOGGER.info(f"{tracker_name}: {params_str}")

    def update(
        self,
        dets: np.ndarray,
        img: np.ndarray,
        embs: np.ndarray = None,
        masks: np.ndarray = None,
    ) -> TrackResults:
        """Update the tracker with new detections for a new frame.

        Handles input preprocessing (unwrapping, layout inference, first-frame
        setup) and per-class splitting automatically.  Subclasses implement
        ``_update_impl`` instead of overriding this method directly.

        Parameters:
        - dets (np.ndarray): Array of detections for the current frame.
        - img (np.ndarray): The current frame as an image array.
        - embs (np.ndarray, optional): Embeddings associated with the detections, if any.
        - masks (np.ndarray, optional): Segmentation masks for the detections,
            shape (N, H, W) where N matches the number of detections.

        Returns:
        - TrackResults: Tracked objects as a numpy subclass with named accessors.
        """
        dets, img = self._preprocess(dets, img)
        masks = self._preprocess_masks(dets, masks)
        result = self._do_update(dets, img, embs, masks)
        if isinstance(result, tuple):
            raw, output_masks = result
        else:
            raw, output_masks = result, None
        return TrackResults(raw, masks=output_masks)

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    def _preprocess(self, dets: np.ndarray, img: np.ndarray):
        """Unwrap inputs and run first-frame setup."""
        # Unwrap `data` attribute if present (e.g. ultralytics tensors)
        if hasattr(dets, "data"):
            dets = dets.data

        # Convert memoryview to numpy array if needed
        if isinstance(dets, memoryview):
            dets = np.array(dets, dtype=np.float32)

        # First-time detection layout inference
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

        # First frame image-based setup (association function needs w/h)
        if not self._first_frame_processed and img is not None:
            self.h, self.w = img.shape[0:2]
            self.asso_func = AssociationFunction(w=self.w, h=self.h, asso_mode=self.asso_func_name).asso_func
            self._first_frame_processed = True

        return dets, img

    def _preprocess_masks(self, dets: np.ndarray, masks: np.ndarray = None) -> np.ndarray:
        """Validate and preprocess segmentation masks."""
        if masks is None:
            return None

        if not self.supports_masks:
            if not getattr(self, "_masks_warning_issued", False):
                LOGGER.warning(f"{self.__class__.__name__} does not support masks. Masks will be ignored.")
                self._masks_warning_issued = True
            return None

        masks = np.asarray(masks)
        if masks.ndim != 3:
            raise ValueError(f"Masks must be 3D (N, H, W), got shape {masks.shape}")

        n_dets = len(dets) if dets is not None else 0
        if masks.shape[0] != n_dets:
            raise ValueError(f"Masks count ({masks.shape[0]}) must match detections count ({n_dets})")

        return masks

    def _do_update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None, masks: np.ndarray = None):
        """Dispatch to single-pass or per-class update."""
        if dets is None or len(dets) == 0:
            dets = self.empty_detections()
            masks = None

        if not self.per_class:
            return self._update_impl(dets=dets, img=img, embs=embs, masks=masks)

        # Per-class splitting
        per_class_tracks = []
        per_class_masks = []
        frame_count = self.frame_count

        for cls_id in range(self.nr_classes):
            class_dets, class_embs = self.get_class_dets_n_embs(dets, embs, cls_id)
            class_masks = self._get_class_masks(dets, masks, cls_id)

            LOGGER.debug(
                f"Processing class {int(cls_id)}: {class_dets.shape} with embeddings"
                f" {class_embs.shape if class_embs is not None else None}"
            )

            # Activate the specific active tracks for this class id
            self.active_tracks = self.per_class_active_tracks[cls_id]
            if hasattr(self, "trackers"):
                self.trackers = self.per_class_trackers[cls_id]

            # Reset frame count for every class
            self.frame_count = frame_count

            # Update detections
            result = self._update_impl(dets=class_dets, img=img, embs=class_embs, masks=class_masks)
            if isinstance(result, tuple):
                tracks, track_masks = result
            else:
                tracks, track_masks = result, None

            # Save the updated active tracks
            self.per_class_active_tracks[cls_id] = self.active_tracks
            if hasattr(self, "trackers"):
                self.per_class_trackers[cls_id] = self.trackers

            if tracks.size > 0:
                per_class_tracks.append(tracks)
                if track_masks is not None:
                    per_class_masks.append(track_masks)

        # Increase frame count by 1
        self.frame_count = frame_count + 1
        if per_class_tracks:
            combined_tracks = np.vstack(per_class_tracks)
            combined_masks = np.vstack(per_class_masks) if per_class_masks else None
            if combined_masks is not None:
                return combined_tracks, combined_masks
            return combined_tracks

        return self.empty_output()

    def _get_class_masks(self, dets: np.ndarray, masks: np.ndarray, cls_id: int):
        """Slice masks by class, matching the logic of get_class_dets_n_embs."""
        if masks is None:
            return None
        if dets.size == 0:
            return None
        class_indices = np.where(dets[:, self.detection_layout.cls_idx] == cls_id)[0]
        if len(class_indices) == 0:
            return None
        return masks[class_indices]

    @abstractmethod
    def _update_impl(
        self,
        dets: np.ndarray,
        img: np.ndarray,
        embs: np.ndarray = None,
        masks: np.ndarray = None,
    ) -> np.ndarray:
        """Core tracking logic. Subclasses must implement this.

        Parameters:
        - dets (np.ndarray): Preprocessed detections for the current frame.
        - img (np.ndarray): The current frame as an image array.
        - embs (np.ndarray, optional): Embeddings associated with the detections.
        - masks (np.ndarray, optional): Segmentation masks for detections, shape (N, H, W).

        Returns:
        - np.ndarray: Raw tracked objects array.
            Mask-capable trackers may return a tuple (tracks_array, output_masks)
            where output_masks has shape (M, H, W) for M active tracks.
        """
        raise NotImplementedError("The _update_impl method needs to be implemented by the subclass.")

    def get_class_dets_n_embs(self, dets, embs, cls_id):
        # Initialize empty arrays for detections and embeddings
        class_dets = self.detection_layout.empty_dets(dtype=np.float32)
        class_embs = np.empty((0, self.last_emb_size)) if self.last_emb_size is not None else None

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
            self.last_emb_size = class_embs.shape[1]  # Update the last known embedding size
        return class_dets, class_embs

    def _set_detection_mode(self, is_obb: bool) -> None:
        """Update the tracker detection mode and association function name."""
        self.detection_layout = get_detection_layout(is_obb)
        self.is_obb = self.detection_layout.is_obb
        self.asso_func_name = self.detection_layout.association_mode_name(self._asso_func_base_name)

        if self._first_frame_processed and hasattr(self, "w") and hasattr(self, "h"):
            self.asso_func = AssociationFunction(w=self.w, h=self.h, asso_mode=self.asso_func_name).asso_func

    def empty_detections(self, dtype=np.float32) -> np.ndarray:
        return self.detection_layout.empty_dets(dtype=dtype)

    def empty_output(self, dtype=float) -> np.ndarray:
        return output_utils.empty_output(self.detection_layout, dtype=dtype)

    def make_detection_batch(
        self,
        dets: np.ndarray,
        embs: np.ndarray | None = None,
        masks: np.ndarray | None = None,
    ) -> DetectionBatch:
        """Convert raw detections to a canonical detection batch."""
        return DetectionBatch.from_layout(
            dets,
            self.detection_layout,
            embs=embs,
            masks=masks,
        )

    def make_detections(
        self,
        dets: np.ndarray,
        embs: np.ndarray | None = None,
        masks: np.ndarray | None = None,
    ) -> list[DetectionRecord]:
        """Convert raw detections to canonical detection records."""
        return self.make_detection_batch(dets, embs=embs, masks=masks).as_records()

    def _track_box_for_output(self, track) -> np.ndarray:
        for attr_name in (
            "output_box",
            "box",
            "bbox",
            "xywha" if self.is_obb else "xyxy",
        ):
            if not attr_name:
                continue
            box = self._resolve_track_box_attr(track, attr_name)
            if box is not None:
                return np.asarray(box, dtype=np.float32).reshape(-1)
        if hasattr(track, "get_state"):
            return np.asarray(track.get_state()[0], dtype=np.float32).reshape(-1)
        raise AttributeError(f"{track.__class__.__name__} does not expose an output box")

    @staticmethod
    def _track_id(track) -> int:
        if hasattr(track, "id"):
            return int(getattr(track, "id"))
        return int(getattr(track, "track_id"))

    def track_record(self, track, state: str = "active") -> TrackRecord:
        """Return a canonical snapshot for a tracker-local track object."""
        return TrackRecord(
            box=self._track_box_for_output(track),
            track_id=self._track_id(track),
            conf=float(getattr(track, "conf", 1.0)),
            cls=int(getattr(track, "cls", -1)),
            det_ind=int(getattr(track, "det_ind", -1)),
            state=state,
            age=int(getattr(track, "age", 0)),
            time_since_update=int(getattr(track, "time_since_update", 0)),
        )

    def format_output_row(
        self,
        box: np.ndarray,
        track_id: int,
        conf: float,
        cls: int,
        det_ind: int,
        dtype=np.float32,
    ) -> np.ndarray:
        """Format one track row using the canonical public output contract."""
        return output_utils.format_output_row(
            self.detection_layout,
            box,
            track_id,
            conf,
            cls,
            det_ind,
            dtype=dtype,
        )

    def format_outputs(self, tracks, dtype=np.float32) -> np.ndarray:
        """Format a sequence of track-like objects into the public output array."""
        rows = [
            self.format_output_row(
                self._track_box_for_output(track),
                self._track_id(track),
                float(getattr(track, "conf", 1.0)),
                int(getattr(track, "cls", -1)),
                int(getattr(track, "det_ind", -1)),
                dtype=dtype,
            )
            for track in tracks
        ]
        return output_utils.format_output_rows(self.detection_layout, rows, dtype=dtype)

    def format_output_rows(self, rows, dtype=np.float32) -> np.ndarray:
        """Return rows with the tracker-specific empty shape when no rows exist."""
        return output_utils.format_output_rows(self.detection_layout, rows, dtype=dtype)

    def cmc_detection_boxes(self, dets: np.ndarray) -> np.ndarray:
        """Return AABB boxes used for camera-motion estimation."""
        return cmc_utils.cmc_detection_boxes(dets, self.detection_layout)

    def aabb_detections_for_association(self, dets: np.ndarray) -> np.ndarray:
        """Return AABB-layout detections for legacy association code."""
        if not self.is_obb:
            return dets

        has_indices = dets.ndim == 2 and dets.shape[1] == self.detection_layout.det_cols + 1
        out_cols = 7 if has_indices else 6
        if dets.size == 0:
            dtype = dets.dtype if hasattr(dets, "dtype") else np.float32
            return np.empty((0, out_cols), dtype=dtype)

        boxes = self.cmc_detection_boxes(dets)
        confs = self.detection_layout.confidences(dets).reshape(-1, 1)
        clss = self.detection_layout.classes(dets).reshape(-1, 1)
        columns = [boxes, confs, clss]
        if has_indices:
            columns.append(dets[:, self.detection_layout.det_cols].reshape(-1, 1))
        return np.hstack(columns).astype(dets.dtype, copy=False)

    def apply_cmc(
        self,
        img: np.ndarray,
        dets: np.ndarray,
        tracks,
        update_method: str = "camera_update",
    ) -> np.ndarray | None:
        """Apply CMC to tracks using OBB-safe detection boxes for estimation."""
        return cmc_utils.apply_cmc_to_tracks(
            getattr(self, "cmc", None),
            img,
            dets,
            self.detection_layout,
            tracks,
            update_method=update_method,
        )

    def filter_outputs_by_geometry(
        self,
        outputs: np.ndarray,
        min_box_area: float | None = None,
        max_aspect_ratio: float | None = None,
    ) -> np.ndarray:
        """Filter output rows by area and width/height ratio in AABB or OBB mode."""
        if outputs.size == 0:
            dtype = outputs.dtype if hasattr(outputs, "dtype") else np.float32
            return self.empty_output(dtype=dtype)

        outputs = np.asarray(outputs)
        if self.is_obb:
            widths = outputs[:, 2]
            heights = outputs[:, 3]
        else:
            widths = outputs[:, 2] - outputs[:, 0]
            heights = outputs[:, 3] - outputs[:, 1]

        keep = np.ones(len(outputs), dtype=bool)
        if max_aspect_ratio is not None:
            keep &= widths / np.maximum(heights, 1e-6) <= max_aspect_ratio
        if min_box_area is not None:
            keep &= widths * heights > min_box_area
        return outputs[keep]

    def check_inputs(self, dets, img, embs=None):
        assert isinstance(dets, np.ndarray), (
            f"Unsupported 'dets' input format '{type(dets)}', valid format is np.ndarray"
        )
        assert isinstance(img, np.ndarray), (
            f"Unsupported 'img_numpy' input format '{type(img)}', valid format is np.ndarray"
        )
        assert len(dets.shape) == 2, "Unsupported 'dets' dimensions, valid number of dimensions is two"

        if embs is not None:
            assert dets.shape[0] == embs.shape[0], "Missmatch between detections and embeddings sizes"

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
        """Map canonical track metadata state onto visualization state names."""
        if state is TrackState.TRACKED:
            return "confirmed"
        if state is TrackState.LOST:
            return "predicted"
        if state is TrackState.REMOVED:
            return "removed"
        return None

    @staticmethod
    def _display_state_from_local_state(track) -> str:
        """Best-effort display mapping for tracker-local state conventions."""
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
        return (getattr(self, "lost_stracks", None) is not None) or (getattr(self, "removed_stracks", None) is not None)

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
        stale_keys = [key for key, first_seen in self._removed_first_seen.items() if first_seen < cutoff]
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

    def _reset_cmc_state(self) -> None:
        """Reset CMC adapters that keep frame-to-frame state."""
        cmc_utils.reset_cmc(getattr(self, "cmc", None))

    def _reset_common_state(self) -> None:
        """Reset BaseTracker-owned sequence state while keeping configuration."""
        self.frame_count = 0
        self.active_tracks = []
        self.last_emb_size = None
        self._first_frame_processed = False
        self._first_dets_processed = False
        self._plot_frame_idx = -1
        self._removed_first_seen.clear()
        self._removed_expired.clear()
        self.id_allocator.reset()

        for attr_name in (
            "trackers",
            "lost_stracks",
            "removed_stracks",
            "lost_tracks",
            "removed_tracks",
        ):
            if hasattr(self, attr_name):
                setattr(self, attr_name, [])

        if self.per_class_active_tracks is not None:
            for cls_id in range(self.nr_classes):
                self.per_class_active_tracks[cls_id] = []
                self.per_class_trackers[cls_id] = []

        self._reset_cmc_state()

    def reset(self):
        """Reset sequence-local tracker state."""
        self._reset_common_state()
