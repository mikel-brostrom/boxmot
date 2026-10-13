"""Runtime Kalman covariance bases for supervised detector/GT calibration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from inspect import Parameter, signature
from typing import Any, get_args, get_type_hints

import numpy as np

from boxmot.structures.kinds import GeometryKind
from boxmot.trackers.common.config import flatten_tracker_options, get_tracker_config_class, load_tracker_config
from boxmot.trackers.common.geometry.obb import align_obb_measurement
from boxmot.trackers.common.mask_guidance import (
    MASK_GUIDANCE_OPTIONS,
    _validated_mask_guidance_values,
)
from boxmot.trackers.common.motion.kalman_filters.config import normalize_kalman_config
from boxmot.trackers.common.motion.kalman_filters.noise import (
    KALMAN_NOISE_OPTIONS,
    KALMAN_TRACKER_NAMES,
    KALMAN_VECTOR_OPTIONS,
)
from boxmot.trackers.common.motion.kalman_filters.profile import (
    calibration_profile_signature,
    validate_calibration_profile,
)
from boxmot.trackers.common.motion.models import MotionModelKind, create_motion_model
from boxmot.trackers.common.registry import get_tracker_class


def validate_calibration_options(tracker_name: str, options: Mapping[str, Any], *, geometry: str = "aabb") -> None:
    """Reject options the runtime tracker cannot consume, without constructing it."""
    tracker_class = get_tracker_class(tracker_name)
    accepted = {
        name
        for name, parameter in signature(tracker_class.__init__).parameters.items()
        if name not in {"self", "config"}
        and parameter.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
    }
    (runtime_options,) = get_args(get_type_hints(tracker_class.__init__)["kwargs"])
    accepted.update(runtime_options.__optional_keys__)
    flattened = flatten_tracker_options(options)
    config_type = get_tracker_config_class(tracker_name)
    algorithm_fields = set(config_type.fields())
    config_type.from_mapping({key: value for key, value in flattened.items() if key in algorithm_fields})
    accepted.update(algorithm_fields)
    validate_calibration_profile(flattened, tracker_name=tracker_name, geometry=geometry)
    accepted.update(name for name in flattened if name.startswith("calibration."))
    if "kalman" in accepted:
        normalize_kalman_config(flattened, tracker_name=tracker_name)
        accepted.update(name for name in flattened if name.startswith("kalman."))
    if "edgetam" in accepted:
        _validated_mask_guidance_values(
            {field: flattened[name] for name, field in MASK_GUIDANCE_OPTIONS.items() if name in flattened}
        )
        accepted.update(MASK_GUIDANCE_OPTIONS)
    unknown = sorted(set(flattened) - accepted)
    if unknown:
        raise ValueError(f"Unsupported {tracker_name} tracker options for KF calibration: {', '.join(unknown)}")


class CalibrationModel:
    """Expose the selected tracker's unscaled P0, R, F(dt), and Q(dt).

    Shared covariance multipliers are reset to one and diagonals cleared when constructing
    the bases. Tracker-specific priors and the selected time units remain in
    effect. No perception models, association, or track lifecycles are run.

    Matrices retain the full runtime state layout. ``measurement_indices``
    and the paired ``velocity_indices``/``velocity_measurement_indices``
    identify coordinates supported by box ground truth. HybridSORT's
    confidence and confidence velocity are deliberately excluded.

    XYHR calibration uses its fixed startup noise policy.
    Online adaptation is outside this reference model. Camera compensation
    uses the shared runtime state transform and its local covariance Jacobian.
    """

    def __init__(
        self,
        tracker_name: str,
        geometry: GeometryKind | str,
        options: Mapping[str, Any],
        *,
        cls_id: int | None = None,
    ) -> None:
        if tracker_name not in KALMAN_TRACKER_NAMES:
            raise ValueError(f"Tracker {tracker_name!r} does not have a supported Kalman model.")
        self.tracker_name = tracker_name
        self.geometry = GeometryKind(geometry)
        self.is_obb = self.geometry is GeometryKind.OBB
        self.options = load_tracker_config(tracker_name, None, options)
        validate_calibration_options(tracker_name, self.options, geometry=self.geometry.value)
        noise = normalize_kalman_config(self.options, tracker_name=tracker_name).noise
        selected_noise = noise.for_class(cls_id) if cls_id is not None else noise
        self.noise_config = replace(
            selected_noise,
            **{name.removeprefix("kalman.noise."): 1.0 for name in KALMAN_NOISE_OPTIONS},
            **{name.removeprefix("kalman.noise."): () for name in KALMAN_VECTOR_OPTIONS},
            by_class={},
        )
        signature_fields = calibration_profile_signature(tracker_name, self.geometry.value, self.options)
        kind = MotionModelKind(signature_fields["filter"])
        self.motion_model = create_motion_model(kind, is_obb=self.is_obb, cls_id=cls_id)
        self.kind = kind
        self.dim_x = self.motion_model.dim_x
        self.dim_z = self.motion_model.dim_z
        self._filter = self.motion_model.create_filter(noise_config=self.noise_config)
        self._matrix_filter = kind in {MotionModelKind.XYSR, MotionModelKind.XYSCR}
        self.motion_model.configure_filter(self._filter)

        self.measurement_indices = tuple(
            index for index in range(self.dim_z) if kind is not MotionModelKind.XYSCR or index != 3
        )
        generator = self._filter._motion_generator()
        pairs = [
            (state_index, measurement_index)
            for state_index in range(self.dim_z, self.dim_x)
            for measurement_index in self.measurement_indices
            if generator[measurement_index, state_index] != 0.0
        ]
        self.velocity_indices = tuple(pair[0] for pair in pairs)
        self.velocity_measurement_indices = tuple(pair[1] for pair in pairs)
        names = {
            MotionModelKind.XYAH: ("x", "y", "aspect_ratio", "height"),
            MotionModelKind.XYWH: ("x", "y", "width", "height"),
            MotionModelKind.XYHR: ("x", "y", "height", "aspect_ratio"),
            MotionModelKind.XYSR: ("x", "y", "area", "aspect_ratio"),
            MotionModelKind.XYSCR: ("x", "y", "area", "confidence", "aspect_ratio"),
        }[kind]
        self.measurement_names = names + (("angle",) if self.is_obb else ())
        self.velocity_names = tuple(
            "v_" + self.measurement_names[int(np.flatnonzero(generator[:, index])[0])]
            for index in range(self.dim_z, self.dim_x)
        )

    def to_measurement(
        self,
        box: np.ndarray,
        score: float = 1.0,
        reference: np.ndarray | None = None,
    ) -> np.ndarray:
        """Convert canonical box geometry, optionally aligning to a prior z."""
        box = np.asarray(box, dtype=float).reshape(-1)
        expected_size = 5 if self.is_obb else 4
        if box.size != expected_size or not np.all(np.isfinite(box)):
            raise ValueError(f"Expected {expected_size} finite values for {self.geometry} box geometry.")
        sizes = box[2:4] if self.is_obb else box[2:4] - box[:2]
        if np.any(sizes <= 0.0):
            raise ValueError("Kalman calibration requires boxes with positive width and height.")
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("Detection confidence must be finite and between zero and one.")
        values = np.r_[box, score] if self.kind is MotionModelKind.XYSCR else box
        measurement = self.motion_model.to_measurement(values, column=False)
        if self.is_obb and reference is not None:
            reference = self._measurement(reference)
            if self.kind in {MotionModelKind.XYWH, MotionModelKind.XYSR, MotionModelKind.XYSCR}:
                measurement = self._filter._align_obb_measurement(measurement, reference)
            else:
                reference_box = self.motion_model.to_box(reference)[0]
                # to_box normally returns a wrapped angle; calibration tracks
                # retain the preceding angle so crossing +/-pi stays smooth.
                angle_index = self.motion_model.angle_measurement_index
                reference_box[4] = reference[angle_index]
                aligned_box = align_obb_measurement(box, reference_box)
                measurement = self.motion_model.to_measurement(aligned_box, column=False)
                measurement[angle_index] = aligned_box[4]
        return measurement

    def _measurement(self, measurement: np.ndarray) -> np.ndarray:
        """Validate a complete measurement vector without altering geometry."""
        values = np.asarray(measurement, dtype=float).reshape(-1)
        if values.size != self.dim_z or not np.all(np.isfinite(values)):
            raise ValueError(f"Expected {self.dim_z} finite Kalman measurement values.")
        return values

    def initial_state(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return runtime's zero-velocity initial mean and base P0."""
        measurement = self._measurement(measurement)
        mean, covariance = self.motion_model.initiate(self._filter, measurement)
        return mean.reshape(-1), covariance

    def measurement_covariance(self, measurement: np.ndarray, score: float = 1.0) -> np.ndarray:
        """Return base R with the same confidence handling used by the tracker.

        Legacy StrongSORT priors can assign zero R to confidence-one
        observations, leaving no positive reference variance for scale fitting.
        """
        measurement = self._measurement(measurement)
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("Detection confidence must be finite and between zero and one.")
        mean = np.zeros(self.dim_x, dtype=float)
        mean[: self.dim_z] = measurement
        covariance = np.zeros((self.dim_x, self.dim_x), dtype=float)
        confidence = score if self.tracker_name == "strongsort" else 0.0
        return self.motion_model.project(self._filter, mean, covariance, confidence=confidence)[1]

    def transition(self, dt: float | None = None) -> np.ndarray:
        """Return discrete fixed-step F or continuous F(dt) in configured units."""
        motion = self._filter.F if self._matrix_filter else self._filter._motion_mat
        return self._filter._elapsed_motion(np.zeros((self.dim_x, self.dim_x)), dt, motion_mat=motion)[0]

    def process_covariance_bases(self, mean: np.ndarray, dt: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return PSD position/velocity bases whose weighted sum is Q(dt).

        Baseline noise is block diagonal. Masking raw position/velocity blocks
        before runtime discretization retains the integrated cross covariance
        induced by velocity noise, including its contribution to position.
        """
        raw = self._raw_process_covariance(mean, dt)
        position = np.zeros_like(raw)
        position[: self.dim_z, : self.dim_z] = raw[: self.dim_z, : self.dim_z]
        velocity = np.zeros_like(raw)
        velocity[self.dim_z :, self.dim_z :] = raw[self.dim_z :, self.dim_z :]
        motion = self._filter.F if self._matrix_filter else self._filter._motion_mat
        return (
            self._filter._elapsed_motion(position, dt, motion_mat=motion)[1],
            self._filter._elapsed_motion(velocity, dt, motion_mat=motion)[1],
        )

    def _raw_process_covariance(self, mean: np.ndarray, dt: float | None) -> np.ndarray:
        """Resolve the native reference Q before coordinate scaling/integration."""
        mean = np.asarray(mean, dtype=float).reshape(-1)
        if mean.size != self.dim_x or not np.all(np.isfinite(mean)):
            raise ValueError(f"Expected {self.dim_x} finite Kalman state values.")
        if self._matrix_filter:
            raw = self._filter.Q
        elif self.kind is MotionModelKind.XYHR:
            raw = self._filter.cov_update_policy.get_q(dt=dt)
        else:
            std_position, std_velocity = self._filter._get_process_noise_std(mean)
            raw = np.diag(np.square(np.r_[std_position, std_velocity]))
        return raw

    def process_coordinate_covariance_bases(self, mean: np.ndarray, dt: float | None = None) -> np.ndarray:
        """Discretize one Q basis per supervised native position/velocity axis.

        Camera rotation can mix axes, so the diagonal multipliers must be
        fitted together after each basis has been propagated into the image.
        Confidence remains outside the supervised coordinate set.
        """
        raw = self._raw_process_covariance(mean, dt)
        if np.any(raw != np.diag(np.diag(raw))):
            raise ValueError("Camera-compensated calibration requires diagonal reference process noise.")
        coordinates = np.asarray(self.measurement_indices + self.velocity_indices, dtype=int)
        bases = np.zeros((len(coordinates), self.dim_x, self.dim_x))
        bases[np.arange(len(coordinates)), coordinates, coordinates] = np.diag(raw)[coordinates]
        motion = self._filter.F if self._matrix_filter else self._filter._motion_mat
        return self._filter._elapsed_motion(bases, dt, motion_mat=motion)[1]

    def camera_transform(self, mean: np.ndarray, warp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the runtime warped mean and its exact covariance Jacobian."""
        transformed, _, jacobian = self.motion_model.transform_states(
            np.asarray(mean).reshape(1, self.dim_x),
            np.zeros((1, self.dim_x, self.dim_x)),
            warp,
            return_jacobian=True,
        )
        transformed = transformed[0]
        if self.is_obb:
            # Runtime wraps angles; GT residuals need the equivalent branch
            # nearest the unwarped prediction rather than a spurious 2*pi jump.
            angle = self.motion_model.angle_measurement_index
            transformed[angle] += 2.0 * np.pi * np.round((mean[angle] - transformed[angle]) / (2.0 * np.pi))
        return transformed, jacobian[0]
