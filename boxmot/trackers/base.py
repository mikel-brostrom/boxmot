from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable, Mapping
from numbers import Real
from pathlib import Path
from typing import Any, overload

import numpy as np
import torch

from boxmot.components.timing import timed_component_phase
from boxmot.motion.kalman_filters.noise import DEFAULT_REFERENCE_DT_S, normalize_kalman_options
from boxmot.structures import (
    Boxes,
    CameraModel,
    Detections,
    Detections3D,
    Frame,
    Geometry,
    MaskBatch,
    MultimodalTracks,
    OrientedBoxes,
    Tracks,
)
from boxmot.trackers.common.appearance.live import _REID_OPTION_UNSET, LiveReIDMixin
from boxmot.trackers.common.association.iou import AssociationFunction
from boxmot.trackers.common.detections import _DetectionBatch
from boxmot.trackers.common.detections.layout import get_detection_layout
from boxmot.trackers.common.geometry.obb import align_obb_measurement
from boxmot.trackers.common.input import (
    frame_image_size,
    pack_numpy_track_rows,
    parse_numpy_detection_rows,
    prepare_frame,
)
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
    LiveReIDMixin,
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
    _requires_detections_3d = False
    _requires_camera = False
    uses_frame_dimensions_for_association = True
    supports_variable_dt = False

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
        variable_dt: bool = False,
        kf_process_position_scale: float = 1.0,
        kf_process_velocity_scale: float = 1.0,
        kf_measurement_noise_scale: float = 1.0,
        kf_initial_position_scale: float = 1.0,
        kf_initial_velocity_scale: float = 1.0,
        kf_reference_dt_s: float = DEFAULT_REFERENCE_DT_S,
        kf_time_unit: str | None = None,
        reid_model: Any | None = _REID_OPTION_UNSET,
        reid_weights: str | Path | list[str | Path] | tuple[str | Path, ...] | None = _REID_OPTION_UNSET,
        device: Any = _REID_OPTION_UNSET,
        half: bool = _REID_OPTION_UNSET,
        reid_preprocess: str | None = _REID_OPTION_UNSET,
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
        - variable_dt: Use measured capture intervals for prediction in seconds.
        - kf_process_position_scale: Multiplier for process noise in measured states.
        - kf_process_velocity_scale: Multiplier for process noise in derivatives.
        - kf_measurement_noise_scale: Multiplier for Kalman measurement covariance.
        - kf_initial_position_scale: Multiplier for initial measured-state covariance.
        - kf_initial_velocity_scale: Multiplier for initial velocity covariance.
        - kf_reference_dt_s: Fixed seconds per reference frame for converting the
          original priors and noise; independent of measured frame intervals.
        - kf_time_unit: Persisted 'frames' or 'seconds', which must agree with
          variable_dt. None derives the unit from the selected timing mode.
        - reid_model: Optional pre-built ReID backend exposing
          ``get_features(boxes, image)``.
        - reid_weights: Weights used to lazily construct a ReID backend when
          an appearance-enabled tracker receives no embeddings.
        - device: Device used by the lazily constructed ReID backend.
        - half: Whether the lazily constructed ReID backend uses FP16.
        - reid_preprocess: Optional ReID preprocessing profile.

        Detection layouts:
        - AABB: ``(x1, y1, x2, y2, conf, cls)``
        - OBB: ``(cx, cy, w, h, angle, conf, cls)``
        """

        if kwargs:
            unexpected = next(iter(kwargs))
            raise TypeError(f"{self.__class__.__name__}.__init__() got an unexpected keyword argument '{unexpected}'")
        if not isinstance(variable_dt, bool):
            raise TypeError("variable_dt must be bool.")
        if variable_dt and not self.supports_variable_dt:
            raise ValueError(f"{self.__class__.__name__} does not support variable_dt.")
        self.variable_dt = variable_dt
        self.kalman_noise_config = normalize_kalman_options(
            {
                "kf_process_position_scale": kf_process_position_scale,
                "kf_process_velocity_scale": kf_process_velocity_scale,
                "kf_measurement_noise_scale": kf_measurement_noise_scale,
                "kf_initial_position_scale": kf_initial_position_scale,
                "kf_initial_velocity_scale": kf_initial_velocity_scale,
                "kf_reference_dt_s": kf_reference_dt_s,
                "kf_time_unit": kf_time_unit,
            },
            variable_dt=variable_dt,
        )
        self.kf_time_unit = self.kalman_noise_config.time_unit
        self.kf_reference_dt_s = self.kalman_noise_config.reference_dt_s
        if not self.kalman_noise_config.is_default and not self.supports_variable_dt:
            raise ValueError(f"{self.__class__.__name__} does not support Kalman noise scaling.")
        self._init_live_reid(
            reid_model=reid_model,
            reid_weights=reid_weights,
            device=device,
            half=half,
            reid_preprocess=reid_preprocess,
        )
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
        self.active_tracks = []
        self.class_track_states = None
        self._first_frame_processed = False
        self._prediction_dt: float | None = None
        self._last_timestamp_s: float | None = None

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
            "variable_dt": self.variable_dt,
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
            detections_3d=bool(self._requires_detections_3d),
            camera=bool(self._requires_camera),
        )

    def _resolve_timing(
        self, frame: Frame | np.ndarray | None, timestamp_s: float | None
    ) -> tuple[float | None, float | None]:
        """Resolve capture time and prediction interval without advancing state."""
        if not self.variable_dt:
            return None, None
        frame_timestamp = frame.timestamp_s if isinstance(frame, Frame) else None
        if timestamp_s is not None and frame_timestamp is not None:
            raise ValueError("Supply timestamp_s either in Frame or as an update argument, not both.")
        timestamp_s = frame_timestamp if timestamp_s is None else timestamp_s
        if timestamp_s is None:
            raise ValueError("variable_dt=True requires timestamp_s on every frame, including the first.")
        if isinstance(timestamp_s, (bool, np.bool_)) or not isinstance(timestamp_s, Real):
            raise ValueError("timestamp_s must be a finite real scalar.")
        try:
            timestamp_s = float(timestamp_s)
        except OverflowError as error:
            raise ValueError("timestamp_s must be a finite real scalar.") from error
        if not np.isfinite(timestamp_s):
            raise ValueError("timestamp_s must be a finite real scalar.")
        dt = None
        if self._last_timestamp_s is not None:
            dt = timestamp_s - self._last_timestamp_s
            if not np.isfinite(dt) or dt <= 0:
                raise ValueError("timestamp_s must increase by a finite positive interval; reset for a new sequence.")
        return timestamp_s, dt

    def validate_timing(
        self, frame: Frame | np.ndarray | None = None, *, timestamp_s: float | None = None
    ) -> float | None:
        """Check capture timing without advancing it, before expensive upstream work."""
        return self._resolve_timing(frame, timestamp_s)[1]

    @overload
    def update(
        self,
        detections: Detections,
        frame: Frame | np.ndarray | None = None,
        *,
        detections_3d: Detections3D,
        camera: CameraModel,
        timestamp_s: float | None = None,
    ) -> MultimodalTracks: ...

    @overload
    def update(
        self, detections: Detections, frame: Frame | np.ndarray | None = None, *, timestamp_s: float | None = None
    ) -> Tracks: ...

    @overload
    def update(
        self, detections: np.ndarray, frame: Frame | np.ndarray | None = None, *, timestamp_s: float | None = None
    ) -> np.ndarray: ...

    def update(
        self,
        detections: Detections | np.ndarray,
        frame: Frame | np.ndarray | None = None,
        *,
        timestamp_s: float | None = None,
        detections_3d: Detections3D | None = None,
        camera: CameraModel | None = None,
    ) -> Tracks | np.ndarray | MultimodalTracks:
        """Advance one sequence with an optional Frame or uint8 HWC BGR image.

        The detection representation determines the output representation.
        With ``variable_dt=True``, derive elapsed seconds from
        ``Frame.timestamp_s`` or the capture timestamp argument. Fixed-step
        prediction is the default even when timestamp metadata is present.
        Calibrated 2D/3D trackers additionally require explicit independent
        ``detections_3d`` and ``camera`` inputs and return ``MultimodalTracks``.
        """
        timestamp_s, dt = self._resolve_timing(frame, timestamp_s)
        frame = prepare_frame(frame)

        if (
            detections_3d is not None
            or camera is not None
            or self.requirements.detections_3d
            or self.requirements.camera
        ):
            return self._update_multimodal(
                detections=detections,
                detections_3d=detections_3d,
                camera=camera,
                frame=frame,
                timestamp_s=timestamp_s,
                dt=dt,
            )

        numpy_input = type(detections) is np.ndarray
        canonical_detections = detections if isinstance(detections, Detections) else None
        mask_image_size = None
        if isinstance(detections, Detections):
            detections.validate()
            self._validate_geometry(detections.geometry)
            if isinstance(frame, Frame) and frame.sample_id != detections.sample_id:
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
            sample_id = None
            geometry = rows.geometry
            scores = rows.scores
            class_ids = rows.class_ids
            embeddings = None
            masks = None

        prepared_bgr = None
        if (
            embeddings is None
            and self.generates_embeddings
            and len(geometry)
            and frame is not None
            and self._reid_encoder_spec is None
        ):
            with timed_component_phase("reid", "preprocess", device=self._reid_device):
                prepared_bgr = self._frame_to_bgr(frame)

        embeddings = self._resolve_input_embeddings(
            geometry=geometry,
            embeddings=embeddings,
            frame=frame,
            detections=canonical_detections,
            scores=scores,
            class_ids=class_ids,
            prepared_bgr=prepared_bgr,
        )
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
        if mask_image_size is not None and frame is not None and mask_image_size != frame_image_size(frame):
            raise ValueError(
                "Detection masks must match the frame spatial size, "
                f"got {mask_image_size} and {frame_image_size(frame)}."
            )

        self._prediction_dt = dt
        self._mark_live_reid_updated()
        tracks = self._update_arrays(
            geometry=geometry,
            scores=scores,
            class_ids=class_ids,
            sample_id=sample_id,
            frame=frame,
            embeddings=embeddings,
            masks=masks,
            mask_image_size=mask_image_size,
            numpy_output=numpy_input,
            prepared_bgr=prepared_bgr,
        )
        self._last_timestamp_s = timestamp_s
        return tracks

    def _update_multimodal(
        self,
        *,
        detections: Detections | np.ndarray,
        detections_3d: Detections3D | None,
        camera: CameraModel | None,
        frame: Frame | np.ndarray | None,
        timestamp_s: float | None,
        dt: float | None,
    ) -> MultimodalTracks:
        """Validate independent sensor inputs before invoking a canonical kernel."""
        capabilities = getattr(self, "capabilities", None)
        requirements = self.requirements
        for name, value in (("detections_3d", detections_3d), ("camera", camera)):
            if value is not None and not getattr(capabilities, f"accepts_{name}", False):
                raise ValueError(f"{self.__class__.__name__} does not accept {name}.")
            if value is None and getattr(requirements, name):
                raise ValueError(f"{self.__class__.__name__} requires {name} on every update.")
        if not isinstance(detections, Detections):
            raise TypeError("Calibrated 2D/3D tracking requires canonical Detections, not packed NumPy rows.")
        if not isinstance(detections_3d, Detections3D):
            raise TypeError("Calibrated 2D/3D tracking requires Detections3D, including explicit empty batches.")
        if not isinstance(camera, CameraModel):
            raise TypeError("Calibrated 2D/3D tracking requires a CameraModel.")
        detections.validate()
        detections_3d.validate()
        camera.validate()
        self._validate_geometry(detections.geometry)
        if detections.sample_id != detections_3d.sample_id:
            raise ValueError("2D and 3D detections must identify the same sample.")
        if isinstance(frame, Frame) and frame.sample_id != detections.sample_id:
            raise ValueError("Frame and detections must identify the same sample.")
        if frame is not None and frame_image_size(frame) != camera.image_size:
            raise ValueError("CameraModel.image_size must match the frame spatial size.")
        if requirements.frame and frame is None:
            raise ValueError(f"{self.__class__.__name__} requires a frame.")
        if requirements.embeddings and detections.embeddings is None:
            raise ValueError(f"{self.__class__.__name__} requires detection embeddings.")
        if requirements.masks and detections.masks is None:
            raise ValueError(f"{self.__class__.__name__} requires full-frame detection masks.")
        if detections.masks is not None:
            if detections.masks.image_size != camera.image_size:
                raise ValueError("Detection masks must match CameraModel.image_size.")
            if requirements.masks and len(detections):
                if not bool(detections.masks.values.flatten(1).any(dim=1).all()):
                    raise ValueError(f"{self.__class__.__name__} requires foreground in every detection mask.")
        self.class_catalog.validate_ids(detections.class_ids.tolist())
        self.class_catalog.validate_ids(detections_3d.class_ids.tolist())

        self._prediction_dt = dt
        self._initialize_frame_dimensions(width=camera.image_size[1], height=camera.image_size[0])
        self._mark_live_reid_updated()
        output = self._track_multimodal(detections, detections_3d, camera, frame)
        if not isinstance(output, MultimodalTracks):
            raise TypeError(f"{self.__class__.__name__} kernel must return MultimodalTracks.")
        output.validate()
        if output.sample_id != detections.sample_id:
            raise ValueError("Multimodal tracker output must identify the current sample.")
        self._validate_geometry(output.image_tracks.geometry)
        for result, observations in (
            (output.image_tracks, detections),
            (output.spatial_tracks, detections_3d),
        ):
            if bool((result.detection_indices >= len(observations)).any()):
                raise ValueError("Multimodal tracker returned a detection index outside its current sensor batch.")
            self.class_catalog.validate_ids(result.class_ids.tolist())
        if output.image_tracks.masks is not None and output.image_tracks.masks.image_size != camera.image_size:
            raise ValueError("Multimodal tracker output masks must match CameraModel.image_size.")
        self._last_timestamp_s = timestamp_s
        return output

    def _track_multimodal(
        self,
        detections: Detections,
        detections_3d: Detections3D,
        camera: CameraModel,
        frame: Frame | np.ndarray | None,
    ) -> MultimodalTracks:
        """Implement calibrated fusion using independently validated sensor rows."""
        raise NotImplementedError(f"{self.__class__.__name__} has no calibrated 2D/3D tracking kernel.")

    def _update_arrays(
        self,
        *,
        geometry: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        sample_id: str | None,
        frame: Frame | np.ndarray | None,
        embeddings: np.ndarray | None,
        masks: np.ndarray | None,
        mask_image_size: tuple[int, int] | None,
        numpy_output: bool,
        prepared_bgr: np.ndarray | None,
    ) -> Tracks | np.ndarray:
        """Invoke one NumPy kernel update and return its requested public representation."""
        self.class_catalog.validate_ids(class_ids.tolist())
        kernel_class_ids = np.asarray(
            [self._encode_kernel_class_id(int(value)) for value in class_ids.tolist()],
            dtype=np.float32,
        )
        dets = np.column_stack((geometry, scores, kernel_class_ids)).astype(np.float32, copy=False)
        requirements = self.requirements
        img = None
        if frame is not None:
            if requirements.frame_dimensions_only or not requirements.frame:
                # A Frame may provide only capture time to a detection-only tracker.
                height, width = frame_image_size(frame)
                self._initialize_frame_dimensions(width=width, height=height)
            else:
                # Tracker kernels and CMC implementations use OpenCV's HWC BGR convention.
                img = prepared_bgr if prepared_bgr is not None else self._frame_to_bgr(frame)

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

        def integer_column(index: int, name: str) -> np.ndarray:
            values = raw[:, index]
            if values.size and not np.equal(values, np.floor(values)).all():
                raise ValueError(f"{self.__class__.__name__} kernel returned non-integer {name}.")
            if values.size and (np.any(values < -(2**63)) or np.any(values >= 2**63)):
                raise ValueError(f"{self.__class__.__name__} kernel returned {name} outside the int64 range.")
            return np.ascontiguousarray(values, dtype=np.int64)

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

        raw_class_ids = integer_column(self.detection_layout.schema.track_class_index, "class IDs")
        decoded_class_ids = np.asarray(
            [self._decode_kernel_class_id(int(value)) for value in raw_class_ids.tolist()],
            dtype=np.int64,
        )
        detection_indices = integer_column(
            self.detection_layout.schema.track_detection_index,
            "detection indices",
        )
        if detection_indices.size and np.any(detection_indices >= len(scores)):
            raise ValueError(f"{self.__class__.__name__} kernel returned a detection index outside the current batch.")
        track_scores = np.ascontiguousarray(
            raw[:, self.detection_layout.schema.track_conf_index],
            dtype=np.float32,
        )

        if numpy_output:
            return pack_numpy_track_rows(
                geometry=geometry_array,
                track_ids=track_ids,
                scores=track_scores,
                class_ids=decoded_class_ids,
                detection_indices=detection_indices,
                is_obb=self.is_obb,
                detection_count=len(scores),
                owner=f"{self.__class__.__name__} kernel",
            )

        geometry_values = torch.from_numpy(geometry_array)
        geometry = OrientedBoxes(geometry_values) if self.is_obb else Boxes(geometry_values)
        assert sample_id is not None

        track_masks = None
        if output_masks is not None:
            output_masks = np.asarray(output_masks)
            if output_masks.ndim != 3 or len(output_masks) != len(raw):
                raise ValueError(f"{self.__class__.__name__} kernel masks must have shape [N, H, W] aligned to tracks.")
            if frame is not None and tuple(output_masks.shape[1:]) != frame_image_size(frame):
                raise ValueError(
                    f"{self.__class__.__name__} kernel masks must match frame size {frame_image_size(frame)}, "
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
            track_ids=torch.from_numpy(track_ids),
            scores=torch.from_numpy(track_scores),
            class_ids=torch.from_numpy(decoded_class_ids),
            detection_indices=torch.from_numpy(detection_indices),
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
        self._prediction_dt = None
        self._last_timestamp_s = None
        self.active_tracks = []
        self._first_frame_processed = False
        self._reset_live_reid_sequence()
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
