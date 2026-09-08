"""Runtime Kalman covariance bases for supervised detector/GT calibration."""

from __future__ import annotations

from collections.abc import Mapping
from inspect import Parameter, signature
from typing import Any

import numpy as np

from boxmot.motion.kalman_filters.noise import (
    KALMAN_NOISE_OPTIONS,
    KALMAN_TRACKER_NAMES,
    normalize_kalman_options,
)
from boxmot.structures.kinds import GeometryKind
from boxmot.trackers.common.geometry.obb import align_obb_measurement
from boxmot.trackers.common.motion import MotionModelKind, create_motion_model
from boxmot.trackers.config import load_tracker_config
from boxmot.trackers.registry import get_tracker_class


def validate_calibration_options(tracker_name: str, options: Mapping[str, Any]) -> None:
    """Reject options the runtime tracker cannot consume, without constructing it."""
    tracker_class = get_tracker_class(tracker_name)
    accepted = {
        name
        for owner in tracker_class.__mro__
        for name, parameter in signature(owner.__init__).parameters.items()
        if name != "self" and parameter.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
    }
    unknown = sorted(set(options) - accepted)
    if unknown:
        raise ValueError(f"Unsupported {tracker_name} tracker options for KF calibration: {', '.join(unknown)}")


class CalibrationModel:
    """Expose the selected tracker's unscaled P0, R, F(dt), and Q(dt).

    The five shared covariance multipliers are reset to one when constructing
    the bases. Tracker-specific priors and the selected time units remain in
    effect. No perception models, association, or track lifecycles are run.

    Matrices retain the full runtime state layout. ``measurement_indices``
    and the paired ``velocity_indices``/``velocity_measurement_indices``
    identify coordinates supported by box ground truth. HybridSORT's
    confidence and confidence velocity are deliberately excluded.

    XYHR calibration uses the fixed startup noise policy. Its optional online
    adaptation and camera compensation are outside this reference model.
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
        validate_calibration_options(tracker_name, self.options)
        self.options.update({name: 1.0 for name in KALMAN_NOISE_OPTIONS})
        self.noise_config = normalize_kalman_options(
            self.options,
            variable_dt=self.options.get("variable_dt", False),
            tracker_name=tracker_name,
        )
        if tracker_name in {"boosttrack", "occluboost"}:
            kind = MotionModelKind.XYHR
        elif tracker_name == "hybridsort" and not self.is_obb:
            kind = MotionModelKind.XYSCR
        elif tracker_name in {"ocsort", "deepocsort", "hybridsort"}:
            kind = MotionModelKind.XYSR
        elif tracker_name == "botsort" or self.is_obb:
            kind = MotionModelKind.XYWH
        else:
            kind = MotionModelKind.XYAH
        self.motion_model = create_motion_model(kind, is_obb=self.is_obb, cls_id=cls_id)
        self.kind = kind
        self.dim_x = self.motion_model.dim_x
        self.dim_z = self.motion_model.dim_z
        self._filter = self.motion_model.create_filter(noise_config=self.noise_config)
        self._matrix_filter = kind in {MotionModelKind.XYSR, MotionModelKind.XYSCR}
        if self._matrix_filter:
            self._configure_sort_priors()

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

    def _configure_sort_priors(self) -> None:
        """Apply the priors installed by SORT-family track constructors."""
        kalman = self._filter
        kalman.R[2:, 2:] *= 10.0
        kalman.P[self.dim_z :, self.dim_z :] *= 1000.0
        kalman.P *= 10.0
        if self.kind is MotionModelKind.XYSCR:
            kalman.Q[-1, -1] *= 0.01
            kalman.Q[-2, -2] *= 0.01
            kalman.Q[self.dim_z :, self.dim_z :] *= 0.01

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
            if self.kind in {MotionModelKind.XYWH, MotionModelKind.XYSR}:
                measurement = self._filter._align_obb_measurement(measurement, reference)
            else:
                reference_box = self.motion_model.to_box(reference)[0]
                # to_box normally returns a wrapped angle; calibration tracks
                # retain the preceding angle so crossing +/-pi stays smooth.
                reference_box[4] = reference[4]
                aligned_box = align_obb_measurement(box, reference_box)
                measurement = self.motion_model.to_measurement(aligned_box, column=False)
                measurement[4] = aligned_box[4]
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
        if self._matrix_filter:
            mean = np.zeros(self.dim_x, dtype=float)
            mean[: self.dim_z] = measurement
            return mean, self._filter.P.copy()
        mean, covariance = self._filter.initiate(measurement)
        return mean.reshape(-1), covariance

    def measurement_covariance(self, measurement: np.ndarray, score: float = 1.0) -> np.ndarray:
        """Return base R with the same confidence handling used by the tracker.

        StrongSORT's confidence-one observations have zero modeled R. Those
        coordinates provide no positive reference variance for scale fitting.
        """
        measurement = self._measurement(measurement)
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("Detection confidence must be finite and between zero and one.")
        if self._matrix_filter:
            return self._filter.noise_config.measurement_covariance(self._filter.R).copy()
        mean = np.zeros(self.dim_x, dtype=float)
        mean[: self.dim_z] = measurement
        covariance = np.zeros((self.dim_x, self.dim_x), dtype=float)
        if self.kind is MotionModelKind.XYHR:
            return self._filter.project(mean, covariance)[1]
        confidence = score if self.tracker_name == "strongsort" else 0.0
        return self._filter.project(mean, covariance, confidence=confidence)[1]

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
        position = np.zeros_like(raw)
        position[: self.dim_z, : self.dim_z] = raw[: self.dim_z, : self.dim_z]
        velocity = np.zeros_like(raw)
        velocity[self.dim_z :, self.dim_z :] = raw[self.dim_z :, self.dim_z :]
        motion = self._filter.F if self._matrix_filter else self._filter._motion_mat
        return (
            self._filter._elapsed_motion(position, dt, motion_mat=motion)[1],
            self._filter._elapsed_motion(velocity, dt, motion_mat=motion)[1],
        )
