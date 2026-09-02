from abc import abstractmethod
from collections.abc import Iterable, Mapping

import numpy as np

from boxmot.trackers.common.association.iou import AssociationFunction
from boxmot.trackers.common.detections import DetectionBatch
from boxmot.trackers.common.detections.layout import get_detection_layout, infer_detection_layout
from boxmot.trackers.common.motion.tracker import TrackerMotionMixin
from boxmot.trackers.common.tracking import outputs as output_utils
from boxmot.trackers.common.tracking.classes import ClassCatalog
from boxmot.trackers.common.tracking.display import TrackDisplayMixin
from boxmot.trackers.common.tracking.formatting import TrackFormattingMixin
from boxmot.trackers.common.tracking.per_class import PerClassUpdateMixin
from boxmot.trackers.common.tracking.records import DetectionRecord
from boxmot.trackers.common.tracking.track import TrackIdAllocator
from boxmot.trackers.common.tracking.visualization import VisualizationMixin
from boxmot.trackers.results import TrackResults
from boxmot.utils import logger as LOGGER


class BaseTracker(
    PerClassUpdateMixin,
    TrackFormattingMixin,
    TrackerMotionMixin,
    TrackDisplayMixin,
    VisualizationMixin,
):
    """Shared public tracker contract.

    ``update`` owns input normalization and output wrapping. Concrete trackers
    implement ``_track_detections`` with their algorithm-specific association and
    lifecycle logic. Centroid association is normalized by frame dimensions, so
    its first update requires an image unless a tracker explicitly opts out of
    dimension-aware association.
    """

    supports_obb = False
    uses_img = False
    uses_embs = False
    supports_masks = False
    uses_frame_dimensions_for_association = True

    def __init__(
        self,
        det_thresh: float = 0.3,
        max_age: int = 30,
        max_obs: int = 50,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        per_class: bool = False,
        class_ids: Iterable[int] | None = None,
        class_names: Mapping[int, str] | None = None,
        asso_func: str = "iou",
        is_obb: bool = False,
        **kwargs,
    ):
        """
        Initialize the BaseTracker object.

        Parameters:
        - det_thresh: Detection threshold for considering detections.
        - max_age: Maximum age in frames before a track is considered lost.
        - max_obs: Maximum number of historical observations stored per track.
        - min_hits: Minimum hits before a track is considered confirmed.
        - iou_threshold: Minimum selected-geometry similarity for matching.
        - per_class: Enable class-separated tracking.
        - class_ids: Optional detector class IDs allowed by this tracker.
        - class_names: Optional detector class names keyed by detector class ID.
        - asso_func: Association geometry: ``iou``, ``giou``, ``diou``,
          ``ciou``, ``hmiou``, or ``centroid`` for AABB and OBB detections.
        - is_obb: Use oriented detections instead of axis-aligned detections.

        Detection layouts:
        - AABB: ``(x1, y1, x2, y2, conf, cls)``
        - OBB: ``(cx, cy, w, h, angle, conf, cls)``
        """

        tracker_name = kwargs.pop("_tracker_name", None)
        self.name = str(tracker_name or self.__class__.__name__)
        self.det_thresh = det_thresh
        self.max_age = max_age
        self.max_obs = max_obs
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.per_class = per_class
        self.class_catalog = ClassCatalog.from_metadata(class_ids=class_ids, class_names=class_names)
        self.class_ids = self.class_catalog.class_ids
        self.class_names = self.class_catalog.names
        if not isinstance(asso_func, str):
            raise TypeError(f"asso_func must be a string, got {type(asso_func).__name__}.")
        self._asso_func_base_name = asso_func.strip().lower()
        if not self._asso_func_base_name:
            raise ValueError("asso_func must not be empty.")
        self.detection_layout = get_detection_layout(is_obb)
        self.asso_func_name = self.detection_layout.association_mode_name(self._asso_func_base_name)
        self.is_obb = self.detection_layout.is_obb
        self.uses_img = bool(
            self.uses_img
            or (self.uses_frame_dimensions_for_association and self.asso_func_name in {"centroid", "centroid_obb"})
        )
        self.asso_func = AssociationFunction(w=None, h=None, asso_mode=self.asso_func_name).asso_func
        self.id_allocator = TrackIdAllocator()

        self.frame_count = 0
        self.active_tracks = []
        self.class_track_states = None
        self._first_frame_processed = False
        self._first_dets_processed = False
        self.last_emb_size = None

        if self.per_class:
            self._initialize_class_track_states()

        if self.max_age >= self.max_obs:
            LOGGER.info("max_age >= max_obs; increasing max_obs to preserve the full track lifetime")
            self.max_obs = self.max_age + 5

        self._plot_frame_idx = -1
        self._removed_first_seen = {}
        self._removed_expired = set()
        self.removed_display_frames = getattr(self, "removed_display_frames", 10)

        if tracker_name:
            base_params = {
                "det_thresh": det_thresh,
                "max_age": max_age,
                "max_obs": max_obs,
                "min_hits": min_hits,
                "iou_threshold": iou_threshold,
                "per_class": per_class,
                "class_ids": None if self.class_ids is None else tuple(sorted(self.class_ids)),
                "asso_func": self._asso_func_base_name,
            }
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
        img: np.ndarray = None,
        embs: np.ndarray = None,
        masks: np.ndarray = None,
    ) -> TrackResults:
        """Update the tracker with one frame of detections."""
        dets, img, embs, masks = self._prepare_update_inputs(
            dets=dets,
            img=img,
            embs=embs,
            masks=masks,
        )
        self._validate_update_inputs(dets=dets, img=img, embs=embs, masks=masks)
        self._initialize_frame_context(img)

        if self.per_class:
            result = self._track_per_class(dets=dets, img=img, embs=embs, masks=masks)
        else:
            result = self._track_detections(dets=dets, img=img, embs=embs, masks=masks)

        if isinstance(result, tuple):
            raw, output_masks = result
        else:
            raw, output_masks = result, None
        return TrackResults(raw, masks=output_masks, schema=self.detection_layout.schema)

    def _prepare_update_inputs(
        self,
        dets: np.ndarray,
        img: np.ndarray = None,
        embs: np.ndarray = None,
        masks: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Unwrap detections and initialize frame context."""
        if hasattr(dets, "dets"):
            if img is None:
                img = getattr(dets, "orig_img", None)
            if masks is None:
                masks = getattr(dets, "masks", None)
            dets = dets.dets

        if hasattr(dets, "data"):
            dets = dets.data

        if isinstance(dets, memoryview):
            dets = np.array(dets, dtype=np.float32)

        if isinstance(dets, np.ndarray) and dets.ndim == 1 and dets.size:
            dets = dets.reshape(1, -1)

        inferred_layout = infer_detection_layout(dets)
        if isinstance(dets, np.ndarray) and dets.ndim == 2 and len(dets) == 0 and inferred_layout is None:
            raise ValueError(
                f"Empty detections must preserve a canonical 6-column AABB or 7-column OBB schema, got {dets.shape}."
            )

        if self._first_dets_processed and inferred_layout is not None:
            if inferred_layout.is_obb != self.detection_layout.is_obb:
                raise ValueError(
                    "Detection modality cannot change after tracker initialization: "
                    f"expected {self.detection_layout.name}, got {inferred_layout.name}."
                )
        elif not self._first_dets_processed and dets is not None:
            if inferred_layout is not None:
                if inferred_layout.is_obb and not self.supports_obb:
                    raise AssertionError(
                        f"{self.__class__.__name__} does not support OBB detections. "
                        "Use an OBB-capable tracker such as ByteTrack, BotSort, OCSort, or SFSORT."
                    )
                self._set_detection_mode(inferred_layout.is_obb)
                self._first_dets_processed = True

        masks = self._prepare_update_masks(dets, masks)
        if dets is None or len(dets) == 0:
            dets = self.empty_detections()
            masks = None
        return dets, img, embs, masks

    def requires_image(
        self,
        dets: np.ndarray,
        embs: np.ndarray | None = None,
        masks: np.ndarray | None = None,
    ) -> bool:
        """Return whether this update needs an image for the active configuration."""
        del dets, embs, masks
        return bool(
            self.uses_frame_dimensions_for_association
            and not self._first_frame_processed
            and self.asso_func_name in {"centroid", "centroid_obb"}
        )

    @staticmethod
    def _requires_live_embeddings(
        dets: np.ndarray,
        embs: np.ndarray | None,
        *,
        enabled: bool,
    ) -> bool:
        """Return whether non-empty detections need image-based ReID extraction."""
        return bool(enabled and embs is None and len(dets) > 0)

    def _initialize_frame_context(self, img: np.ndarray | None) -> None:
        """Record frame dimensions and bind dimension-aware association once."""
        if self._first_frame_processed or img is None:
            return
        self._initialize_frame_dimensions(width=img.shape[1], height=img.shape[0])

    def _initialize_frame_dimensions(self, *, width: int, height: int) -> None:
        """Bind dimension-aware association without requiring an image buffer."""
        if self._first_frame_processed:
            return
        if width <= 0 or height <= 0:
            raise ValueError(f"Frame dimensions must be positive, got width={width}, height={height}.")
        self.w, self.h = int(width), int(height)
        self.asso_func = AssociationFunction(w=self.w, h=self.h, asso_mode=self.asso_func_name).asso_func
        self._first_frame_processed = True

    def association_similarity(
        self,
        boxes_a: np.ndarray | Iterable[object],
        boxes_b: np.ndarray | Iterable[object],
    ) -> np.ndarray:
        """Return the configured geometric similarity for two box collections."""
        geometry_a = self._association_boxes(boxes_a)
        geometry_b = self._association_boxes(boxes_b)
        if len(geometry_a) == 0 or len(geometry_b) == 0:
            return np.empty((len(geometry_a), len(geometry_b)), dtype=np.float32)
        return np.asarray(self.asso_func(geometry_a, geometry_b))

    def association_distance(
        self,
        atracks: np.ndarray | Iterable[object],
        btracks: np.ndarray | Iterable[object],
    ) -> np.ndarray:
        """Return ``1 - similarity`` for arrays or track-like objects."""
        return 1.0 - self.association_similarity(atracks, btracks)

    def _association_boxes(self, items: np.ndarray | Iterable[object]) -> np.ndarray:
        """Extract canonical AABB/OBB geometry from arrays or track-like objects."""
        geometry_cols = self.detection_layout.box_cols
        if isinstance(items, np.ndarray):
            values = np.asarray(items)
            if values.ndim == 1:
                if values.size == 0:
                    return np.empty((0, geometry_cols), dtype=values.dtype)
                values = values.reshape(1, -1)
            if values.ndim == 2 and len(values) == 0:
                return np.empty((0, geometry_cols), dtype=values.dtype)
            if values.ndim != 2 or values.shape[1] < geometry_cols:
                raise ValueError(
                    f"Association boxes must be a 2D array with at least {geometry_cols} columns, "
                    f"got shape {values.shape}."
                )
            return values[:, :geometry_cols]

        values = list(items)
        if not values:
            return np.empty((0, geometry_cols), dtype=np.float32)

        geometry_attr = "xywha" if self.is_obb else "xyxy"
        rows = []
        for item in values:
            if isinstance(item, np.ndarray):
                row = np.asarray(item).reshape(-1)
            else:
                try:
                    row = np.asarray(getattr(item, geometry_attr)).reshape(-1)
                except AttributeError as exc:
                    raise TypeError(
                        f"Association item {type(item).__name__} must expose {geometry_attr!r} geometry."
                    ) from exc
            if row.size < geometry_cols:
                raise ValueError(
                    f"Association item must provide at least {geometry_cols} geometry values, got {row.size}."
                )
            rows.append(row[:geometry_cols])
        return np.asarray(rows)

    def _prepare_update_masks(self, dets: np.ndarray, masks: np.ndarray = None) -> np.ndarray | None:
        """Normalize optional masks and discard them for unsupported trackers."""
        if masks is None:
            return None

        if not self.supports_masks:
            if not getattr(self, "_masks_warning_issued", False):
                LOGGER.warning(f"{self.__class__.__name__} does not support masks. Masks will be ignored.")
                self._masks_warning_issued = True
            return None

        return np.asarray(masks)

    def _validate_update_inputs(
        self,
        dets: np.ndarray,
        img: np.ndarray | None = None,
        embs: np.ndarray | None = None,
        masks: np.ndarray | None = None,
    ) -> None:
        """Validate canonical detections and optional frame-aligned inputs."""
        self.detection_layout.validate_dets(dets)
        if dets.size and not np.isfinite(dets).all():
            raise ValueError("Tracker detections must contain only finite values.")
        if dets.size:
            boxes = self.detection_layout.boxes(dets)
            if self.is_obb:
                if np.any(boxes[:, 2:4] <= 0):
                    raise ValueError("OBB detections must have positive width and height.")
            elif np.any(boxes[:, 2] <= boxes[:, 0]) or np.any(boxes[:, 3] <= boxes[:, 1]):
                raise ValueError("AABB detections must satisfy x2 > x1 and y2 > y1.")
        self.class_catalog.validate_detections(dets, self.detection_layout)

        if img is not None:
            if not isinstance(img, np.ndarray):
                raise TypeError(f"Unsupported image type {type(img).__name__}; expected numpy.ndarray.")
            if img.ndim not in (2, 3):
                raise ValueError(f"Image must be a 2D or 3D array, got shape {img.shape}.")
            if img.shape[0] == 0 or img.shape[1] == 0:
                raise ValueError(f"Image must have non-zero height and width, got shape {img.shape}.")

        if embs is not None:
            if not isinstance(embs, np.ndarray):
                raise TypeError(f"Unsupported embeddings type {type(embs).__name__}; expected numpy.ndarray.")
            if embs.ndim != 2:
                raise ValueError(f"Embeddings must be a 2D array, got shape {embs.shape}.")
            if len(embs) != len(dets):
                raise ValueError("Detections and embeddings must have the same number of rows.")
            if embs.size and not np.isfinite(embs).all():
                raise ValueError("Embeddings must contain only finite values.")

        if img is None and self.requires_image(dets=dets, embs=embs, masks=masks):
            if (
                self.uses_frame_dimensions_for_association
                and not self._first_frame_processed
                and self.asso_func_name in {"centroid", "centroid_obb"}
            ):
                raise ValueError(
                    f"{self.__class__.__name__} requires img when using '{self._asso_func_base_name}' association."
                )
            raise ValueError(f"{self.__class__.__name__} requires img for the current tracker configuration.")

        if masks is None:
            return
        if masks.ndim != 3:
            raise ValueError(f"Masks must be 3D (N, H, W), got shape {masks.shape}")

        n_dets = len(dets)
        if masks.shape[0] != n_dets:
            raise ValueError(f"Masks count ({masks.shape[0]}) must match detections count ({n_dets})")

    @abstractmethod
    def _track_detections(
        self,
        dets: np.ndarray,
        img: np.ndarray | None,
        embs: np.ndarray = None,
        masks: np.ndarray = None,
    ) -> np.ndarray:
        """Run algorithm-specific tracking for one frame."""
        raise NotImplementedError("The _track_detections method needs to be implemented by the subclass.")

    def _set_detection_mode(self, is_obb: bool) -> None:
        """Update detection layout and association function mode."""
        self.detection_layout = get_detection_layout(is_obb)
        self.is_obb = self.detection_layout.is_obb
        self.asso_func_name = self.detection_layout.association_mode_name(self._asso_func_base_name)
        if self.uses_frame_dimensions_for_association and self.asso_func_name in {"centroid", "centroid_obb"}:
            self.uses_img = True

        if self._first_frame_processed and hasattr(self, "w") and hasattr(self, "h"):
            self.asso_func = AssociationFunction(w=self.w, h=self.h, asso_mode=self.asso_func_name).asso_func
        else:
            self.asso_func = AssociationFunction(w=None, h=None, asso_mode=self.asso_func_name).asso_func

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

    def configure_class_catalog(
        self,
        *,
        class_ids: Iterable[int] | None = None,
        class_names: Mapping[int, str] | None = None,
    ) -> None:
        """Replace detector class metadata before a new sequence starts."""
        self.class_catalog = ClassCatalog.from_metadata(class_ids=class_ids, class_names=class_names)
        self.class_ids = self.class_catalog.class_ids
        self.class_names = self.class_catalog.names

    def _reset_common_state(self) -> None:
        """Reset sequence-local state while keeping tracker configuration."""
        self.frame_count = 0
        self.active_tracks = []
        self.last_emb_size = None
        self._first_frame_processed = False
        self._first_dets_processed = False
        self._plot_frame_idx = -1
        self._removed_first_seen.clear()
        self._removed_expired.clear()
        self.id_allocator.reset()

        for attr_name in self._class_state_attr_names():
            if hasattr(self, attr_name):
                setattr(self, attr_name, self._empty_state_like(getattr(self, attr_name)))

        self._reset_class_track_states()
        self._reset_cmc_state()

    def reset(self):
        """Reset sequence-local tracker state."""
        self._reset_common_state()
