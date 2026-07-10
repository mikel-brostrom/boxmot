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
    implement ``_update_impl`` with their algorithm-specific association and
    lifecycle logic.
    """

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
        - iou_threshold: IoU threshold for detection-track matching.
        - per_class: Enable class-separated tracking.
        - class_ids: Optional detector class IDs allowed by this tracker.
        - class_names: Optional detector class names keyed by detector class ID.
        - asso_func: Association function name.
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
        self._asso_func_base_name = asso_func
        self.detection_layout = get_detection_layout(is_obb)
        self.asso_func_name = self.detection_layout.association_mode_name(asso_func)
        self.is_obb = self.detection_layout.is_obb
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
            LOGGER.warning("Max age > max observations, increasing size of max observations...")
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
                "asso_func": asso_func,
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
        *,
        image: np.ndarray = None,
        embeddings: np.ndarray = None,
    ) -> TrackResults:
        """Update the tracker with one frame of detections."""
        if image is not None:
            if img is not None:
                raise ValueError("Use only one of img=... or image=...")
            img = image
        if embeddings is not None:
            if embs is not None:
                raise ValueError("Use only one of embs=... or embeddings=...")
            embs = embeddings

        if hasattr(dets, "dets"):
            if img is None:
                img = getattr(dets, "orig_img", None)
            if masks is None:
                masks = getattr(dets, "masks", None)
            dets = dets.dets

        dets, img = self._preprocess(dets, img)
        masks = self._preprocess_masks(dets, masks)
        result = self._do_update(dets, img, embs, masks)
        if isinstance(result, tuple):
            raw, output_masks = result
        else:
            raw, output_masks = result, None
        return TrackResults(raw, masks=output_masks)

    def _preprocess(self, dets: np.ndarray, img: np.ndarray):
        """Unwrap inputs, infer detection layout, and initialize frame context."""
        if hasattr(dets, "data"):
            dets = dets.data

        if isinstance(dets, memoryview):
            dets = np.array(dets, dtype=np.float32)

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

        if not self._first_frame_processed and img is not None:
            self.h, self.w = img.shape[0:2]
            self.asso_func = AssociationFunction(w=self.w, h=self.h, asso_mode=self.asso_func_name).asso_func
            self._first_frame_processed = True

        return dets, img

    def _preprocess_masks(self, dets: np.ndarray, masks: np.ndarray = None) -> np.ndarray:
        """Validate optional segmentation masks."""
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
        """Dispatch to single-class or class-separated tracking."""
        if dets is None or len(dets) == 0:
            dets = self.empty_detections()
            masks = None

        self.detection_layout.validate_dets(dets)
        self.class_catalog.validate_detections(dets, self.detection_layout)

        if not self.per_class:
            return self._update_impl(dets=dets, img=img, embs=embs, masks=masks)

        return self._update_per_class(dets=dets, img=img, embs=embs, masks=masks)

    @abstractmethod
    def _update_impl(
        self,
        dets: np.ndarray,
        img: np.ndarray,
        embs: np.ndarray = None,
        masks: np.ndarray = None,
    ) -> np.ndarray:
        """Run algorithm-specific tracking for one frame."""
        raise NotImplementedError("The _update_impl method needs to be implemented by the subclass.")

    def _set_detection_mode(self, is_obb: bool) -> None:
        """Update detection layout and association function mode."""
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

    def check_inputs(self, dets, img, embs=None):
        assert isinstance(dets, np.ndarray), (
            f"Unsupported 'dets' input format '{type(dets)}', valid format is np.ndarray"
        )
        assert isinstance(img, np.ndarray), (
            f"Unsupported 'img_numpy' input format '{type(img)}', valid format is np.ndarray"
        )
        assert len(dets.shape) == 2, "Unsupported 'dets' dimensions, valid number of dimensions is two"

        if embs is not None:
            assert dets.shape[0] == embs.shape[0], "Mismatch between detections and embeddings sizes"

        self.detection_layout.validate_dets(dets)

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
