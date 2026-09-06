from abc import abstractmethod
from collections.abc import Iterable, Mapping

import numpy as np
import torch

from boxmot.structures import Boxes, Detections, Frame, Geometry, MaskBatch, OrientedBoxes, Tracks
from boxmot.trackers.common.association.iou import AssociationFunction
from boxmot.trackers.common.detections import _DetectionBatch
from boxmot.trackers.common.detections.layout import get_detection_layout
from boxmot.trackers.common.geometry.obb import align_obb_measurement
from boxmot.trackers.common.input import parse_numpy_detection_rows
from boxmot.trackers.common.motion.tracker import TrackerMotionMixin
from boxmot.trackers.common.tracking import outputs as output_utils
from boxmot.trackers.common.tracking.classes import ClassCatalog
from boxmot.trackers.common.tracking.display import TrackDisplayMixin
from boxmot.trackers.common.tracking.formatting import TrackFormattingMixin
from boxmot.trackers.common.tracking.per_class import PerClassUpdateMixin
from boxmot.trackers.common.tracking.track import TrackIdAllocator
from boxmot.trackers.common.tracking.visualization import VisualizationMixin
from boxmot.trackers.protocols import TrackerRequirements
from boxmot.utils import logger as LOGGER


class BaseTracker(
    PerClassUpdateMixin,
    TrackFormattingMixin,
    TrackerMotionMixin,
    TrackDisplayMixin,
    VisualizationMixin,
):
    """Shared public tracker contract.

    ``update`` owns public input validation and canonical output wrapping. Concrete
    trackers implement ``_track_detections`` with their algorithm-specific
    association and lifecycle logic. Centroid association is normalized by frame
    dimensions, so its first update requires an image unless a tracker explicitly
    opts out of dimension-aware association.
    """

    supports_obb = False
    use_embeddings = False
    _requires_frame = False
    _requires_frame_dimensions_only = False
    _requires_masks = False
    uses_frame_dimensions_for_association = True

    def _resolve_detection_layout(self, is_obb: bool):
        """Return the private row layout for a resolved geometry mode."""

        return get_detection_layout(is_obb)

    def _resolve_association_mode_name(self, base_name: str) -> str:
        """Resolve an association name for the configured geometry layout."""

        return self.detection_layout.association_mode_name(base_name)

    def _build_association_function(self, *, width: int | None, height: int | None):
        """Build the selected association callable for optional frame dimensions."""

        return AssociationFunction(w=width, h=height, asso_mode=self.asso_func_name).asso_func

    def _validate_geometry(self, geometry: Geometry) -> None:
        """Validate canonical geometry against this tracker's fixed mode."""

        expected_obb = self.detection_layout.is_obb
        if geometry.is_obb != expected_obb:
            expected = "OBB" if expected_obb else "AABB"
            received = "OBB" if geometry.is_obb else "AABB"
            raise ValueError(f"{self.__class__.__name__} is configured for {expected} geometry, got {received}.")

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
          OBB ``ciou`` is a custom experimental long/short-side adaptation.
          OBB ``hmiou`` is an experimental global-y height cue intended only
          where image vertical is meaningful.
        - is_obb: Use oriented detections instead of axis-aligned detections.

        Detection layouts:
        - AABB: ``(x1, y1, x2, y2, conf, cls)``
        - OBB: ``(cx, cy, w, h, angle, conf, cls)``
        """

        if kwargs:
            unexpected = next(iter(kwargs))
            raise TypeError(f"{self.__class__.__name__}.__init__() got an unexpected keyword argument '{unexpected}'")

        self.name = self.__class__.__name__
        self.det_thresh = det_thresh
        self.max_age = max_age
        self.max_obs = max_obs
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.per_class = per_class
        self.class_catalog = ClassCatalog.from_metadata(class_ids=class_ids, class_names=class_names)
        self.class_ids = self.class_catalog.class_ids
        self.class_names = self.class_catalog.names
        # Legacy kernels store geometry and metadata in one float32 row. Keep
        # canonical int64 class IDs outside that row and expose only stable,
        # exactly representable private codes to the kernels.
        self._canonical_to_kernel_class_id: dict[int, int] = {}
        self._kernel_to_canonical_class_id: dict[int, int] = {}
        self._next_kernel_class_id = -1
        self._obb_output_by_track_id: dict[int, np.ndarray] = {}
        if not isinstance(asso_func, str):
            raise TypeError(f"asso_func must be a string, got {type(asso_func).__name__}.")
        if not asso_func:
            raise ValueError("asso_func must not be empty.")
        if asso_func != asso_func.strip().lower():
            raise ValueError(f"asso_func must use its canonical lowercase identifier: {asso_func!r}")
        self._asso_func_base_name = asso_func
        self.detection_layout = self._resolve_detection_layout(is_obb)
        if self.detection_layout.is_obb and not self.supports_obb:
            raise ValueError(f"{self.__class__.__name__} does not support OBB geometry.")
        self.asso_func_name = self._resolve_association_mode_name(self._asso_func_base_name)
        self.is_obb = self.detection_layout.is_obb
        self._requires_frame = bool(
            self._requires_frame
            or (self.uses_frame_dimensions_for_association and self.asso_func_name in {"centroid", "centroid_obb"})
        )
        self.asso_func = self._build_association_function(width=None, height=None)
        self.id_allocator = TrackIdAllocator()

        self.frame_count = 0
        self._numpy_sample_index = 0
        self.active_tracks = []
        self.class_track_states = None
        self._first_frame_processed = False
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
        params_str = ", ".join(f"{k}={v}" for k, v in base_params.items())
        LOGGER.info(f"{self.name}: {params_str}")

    @property
    def requirements(self) -> TrackerRequirements:
        """Return the immutable inputs required by this resolved configuration."""
        return TrackerRequirements(
            embeddings=bool(self.use_embeddings),
            masks=bool(self._requires_masks),
            frame=bool(self._requires_frame),
            frame_dimensions_only=bool(self._requires_frame_dimensions_only),
        )

    def update(self, detections: Detections | np.ndarray, frame: Frame | None = None) -> Tracks:
        """Advance one sequence using canonical structures or packed NumPy rows."""
        if frame is not None and not isinstance(frame, Frame):
            raise TypeError(f"frame must be Frame or None, got {type(frame).__name__}.")
        if frame is not None:
            frame.validate()

        numpy_input = isinstance(detections, np.ndarray)
        mask_image_size = None
        if isinstance(detections, Detections):
            detections.validate()
            self._validate_geometry(detections.geometry)
            if frame is not None and frame.sample_id != detections.sample_id:
                raise ValueError(
                    "Frame and detections must identify the same sample, "
                    f"got {frame.sample_id!r} and {detections.sample_id!r}."
                )
            sample_id = detections.sample_id
            geometry = detections.geometry.values.detach().numpy()
            scores = detections.scores.detach().numpy()
            class_ids = detections.class_ids.detach().numpy()
            embeddings = None if detections.embeddings is None else detections.embeddings.detach().numpy()
            masks = None if detections.masks is None else detections.masks.values.detach().numpy()
            if detections.masks is not None:
                mask_image_size = detections.masks.image_size
        else:
            rows = parse_numpy_detection_rows(detections, is_obb=self.is_obb)
            sample_id = frame.sample_id if frame is not None else f"numpy:{self._numpy_sample_index:06d}"
            geometry = rows.geometry
            scores = rows.scores
            class_ids = rows.class_ids
            embeddings = None
            masks = None

        requirements = self.requirements
        if requirements.embeddings and embeddings is None:
            raise ValueError(f"{self.__class__.__name__} requires detection embeddings.")
        if requirements.masks and masks is None:
            raise ValueError(f"{self.__class__.__name__} requires full-frame detection masks.")
        if requirements.masks and len(scores):
            assert masks is not None
            if not masks.reshape(len(scores), -1).any(axis=1).all():
                raise ValueError(f"{self.__class__.__name__} requires foreground in every non-empty detection mask.")
        if requirements.frame and frame is None:
            raise ValueError(f"{self.__class__.__name__} requires a frame.")
        if mask_image_size is not None and frame is not None and mask_image_size != frame.image_size:
            raise ValueError(
                f"Detection masks must match the frame spatial size, got {mask_image_size} and {frame.image_size}."
            )

        tracks = self._update_arrays(
            geometry=geometry,
            scores=scores,
            class_ids=class_ids,
            sample_id=sample_id,
            frame=frame,
            embeddings=embeddings,
            masks=masks,
            mask_image_size=mask_image_size,
        )
        if numpy_input and frame is None:
            self._numpy_sample_index += 1
        return tracks

    def _update_arrays(
        self,
        *,
        geometry: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        sample_id: str,
        frame: Frame | None,
        embeddings: np.ndarray | None,
        masks: np.ndarray | None,
        mask_image_size: tuple[int, int] | None,
    ) -> Tracks:
        """Invoke one legacy NumPy kernel update and wrap its typed result."""
        self.class_catalog.validate_ids(class_ids.tolist())
        kernel_class_ids = np.asarray(
            [self._encode_kernel_class_id(int(value)) for value in class_ids.tolist()],
            dtype=np.float32,
        )
        dets = np.column_stack((geometry, scores, kernel_class_ids)).astype(np.float32, copy=False)
        requirements = self.requirements
        img = None
        if frame is not None:
            if requirements.frame_dimensions_only:
                self._initialize_frame_dimensions(width=frame.width, height=frame.height)
            else:
                # Tracker kernels and CMC implementations use OpenCV's HWC BGR convention.
                img = frame.image.permute(1, 2, 0).flip(-1).contiguous().numpy()

        self._initialize_frame_context(img)
        if self.per_class:
            result = self._track_per_class(dets=dets, img=img, embs=embeddings, masks=masks)
        else:
            result = self._track_detections(dets=dets, img=img, embs=embeddings, masks=masks)

        if isinstance(result, tuple):
            raw, output_masks = result
        else:
            raw, output_masks = result, None

        raw = np.asarray(raw, dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] != self.detection_layout.output_cols:
            raise ValueError(
                f"{self.__class__.__name__} kernel returned shape {raw.shape}; "
                f"expected [N, {self.detection_layout.output_cols}]."
            )
        if raw.size and not np.isfinite(raw).all():
            raise ValueError(f"{self.__class__.__name__} kernel returned non-finite track rows.")

        def integer_column(index: int, name: str) -> torch.Tensor:
            values = raw[:, index]
            if values.size and not np.equal(values, np.floor(values)).all():
                raise ValueError(f"{self.__class__.__name__} kernel returned non-integer {name}.")
            return torch.from_numpy(np.ascontiguousarray(values, dtype=np.int64))

        track_ids = integer_column(self.detection_layout.schema.track_id_index, "track IDs")
        geometry_columns = self.detection_layout.box_cols
        geometry_array = np.ascontiguousarray(raw[:, :geometry_columns], dtype=np.float32)
        if self.is_obb:
            geometry_array = geometry_array.copy()
            for index, track_id in enumerate(track_ids.tolist()):
                previous = self._obb_output_by_track_id.get(track_id)
                if previous is not None:
                    geometry_array[index] = align_obb_measurement(geometry_array[index], previous)
                self._obb_output_by_track_id[track_id] = geometry_array[index].copy()
        geometry_values = torch.from_numpy(geometry_array)
        geometry = OrientedBoxes(geometry_values) if self.is_obb else Boxes(geometry_values)

        raw_class_ids = integer_column(self.detection_layout.schema.track_class_index, "class IDs")
        decoded_class_ids = torch.tensor(
            [self._decode_kernel_class_id(int(value)) for value in raw_class_ids.tolist()],
            dtype=torch.int64,
        )
        detection_indices = integer_column(
            self.detection_layout.schema.track_detection_index,
            "detection indices",
        )
        if detection_indices.numel() and bool((detection_indices >= len(scores)).any()):
            raise ValueError(f"{self.__class__.__name__} kernel returned a detection index outside the current batch.")

        track_masks = None
        if output_masks is not None:
            output_masks = np.asarray(output_masks)
            if output_masks.ndim != 3 or len(output_masks) != len(raw):
                raise ValueError(f"{self.__class__.__name__} kernel masks must have shape [N, H, W] aligned to tracks.")
            if frame is not None and tuple(output_masks.shape[1:]) != frame.image_size:
                raise ValueError(
                    f"{self.__class__.__name__} kernel masks must match frame size {frame.image_size}, "
                    f"got {tuple(output_masks.shape[1:])}."
                )
            mask_values = torch.from_numpy(np.ascontiguousarray(output_masks, dtype=np.bool_))
            track_masks = MaskBatch(mask_values)
        elif self.requirements.masks and len(raw) == 0:
            assert mask_image_size is not None
            height, width = mask_image_size
            track_masks = MaskBatch(torch.empty((0, height, width), dtype=torch.bool))
        elif self.requirements.masks:
            raise ValueError(f"{self.__class__.__name__} must return track-aligned masks.")

        return Tracks(
            geometry=geometry,
            track_ids=track_ids,
            scores=torch.from_numpy(
                np.ascontiguousarray(raw[:, self.detection_layout.schema.track_conf_index], dtype=np.float32)
            ),
            class_ids=decoded_class_ids,
            detection_indices=detection_indices,
            sample_id=sample_id,
            masks=track_masks,
        )

    def _encode_kernel_class_id(self, class_id: int) -> int:
        """Map a canonical ID to a stable float32/int32-safe kernel code."""
        existing = self._canonical_to_kernel_class_id.get(class_id)
        if existing is not None:
            return existing

        int32_max = np.iinfo(np.int32).max
        if class_id <= int32_max and int(np.float32(class_id)) == class_id:
            kernel_id = class_id
        else:
            kernel_id = self._next_kernel_class_id
            self._next_kernel_class_id -= 1
        self._canonical_to_kernel_class_id[class_id] = kernel_id
        self._kernel_to_canonical_class_id[kernel_id] = class_id
        return kernel_id

    def _decode_kernel_class_id(self, kernel_id: int) -> int:
        """Restore one exact canonical class ID emitted by a private kernel."""
        try:
            return self._kernel_to_canonical_class_id[kernel_id]
        except KeyError as exc:
            raise ValueError(f"{self.__class__.__name__} kernel returned unknown class code {kernel_id}.") from exc

    def _validate_kernel_class_ids(self, kernel_ids: Iterable[int]) -> None:
        """Validate private class codes against the public class catalog."""
        self.class_catalog.validate_ids(self._decode_kernel_class_id(int(value)) for value in kernel_ids)

    def _kernel_class_id_for_lookup(self, class_id: int) -> int:
        """Resolve a public class ID for class-local state inspection."""
        self.class_catalog.validate_ids((class_id,))
        return self._encode_kernel_class_id(int(class_id))

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
        self.asso_func = self._build_association_function(width=self.w, height=self.h)
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

    def _empty_output(self, dtype=float) -> np.ndarray:
        """Return the private NumPy kernel's empty output layout."""
        return output_utils.empty_output(self.detection_layout, dtype=dtype)

    def _make_detection_batch(
        self,
        dets: np.ndarray,
        embs: np.ndarray | None = None,
        masks: np.ndarray | None = None,
    ) -> _DetectionBatch:
        """Convert raw detections to a canonical detection batch."""
        return _DetectionBatch.from_layout(
            dets,
            self.detection_layout,
            embs=embs,
            masks=masks,
        )

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
        self._numpy_sample_index = 0
        self.active_tracks = []
        self.last_emb_size = None
        self._first_frame_processed = False
        self._plot_frame_idx = -1
        self._removed_first_seen.clear()
        self._removed_expired.clear()
        self.id_allocator.reset()
        self._canonical_to_kernel_class_id.clear()
        self._kernel_to_canonical_class_id.clear()
        self._next_kernel_class_id = -1
        self._obb_output_by_track_id.clear()

        for attr_name in self._class_state_attr_names():
            if hasattr(self, attr_name):
                setattr(self, attr_name, self._empty_state_like(getattr(self, attr_name)))

        self._reset_class_track_states()
        self._reset_cmc_state()

    def reset(self):
        """Reset sequence-local tracker state."""
        self._reset_common_state()
