"""Immutable, instance-local Kalman covariance calibration."""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite, sqrt
from numbers import Real

import numpy as np

KALMAN_TRACKER_NAMES = frozenset(
    {"botsort", "boosttrack", "bytetrack", "deepocsort", "hybridsort", "occluboost", "ocsort", "strongsort"}
)
KALMAN_NOISE_OPTIONS = (
    "kf_process_position_scale",
    "kf_process_velocity_scale",
    "kf_measurement_noise_scale",
    "kf_initial_position_scale",
    "kf_initial_velocity_scale",
)
KALMAN_TIMING_OPTIONS = ("kf_time_unit", "kf_reference_dt_s")
DEFAULT_REFERENCE_DT_S = 1.0 / 30.0


@dataclass(frozen=True, slots=True)
class KalmanNoiseConfig:
    """Covariance multipliers and a fixed conversion from frame priors to seconds.

    Position includes every measured state (size, ratio, angle, or confidence),
    while velocity includes their modeled derivatives. The reference interval
    describes the original frame-based priors; it never follows observed dt.
    """

    process_position_scale: float = 1.0
    process_velocity_scale: float = 1.0
    measurement_noise_scale: float = 1.0
    initial_position_scale: float = 1.0
    initial_velocity_scale: float = 1.0
    time_unit: str = "frames"
    reference_dt_s: float = DEFAULT_REFERENCE_DT_S

    def __post_init__(self) -> None:
        if not isinstance(self.time_unit, str) or self.time_unit not in ("frames", "seconds"):
            raise ValueError("kf_time_unit must be 'frames' or 'seconds'.")
        for name in (*[option.removeprefix("kf_") for option in KALMAN_NOISE_OPTIONS], "reference_dt_s"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise ValueError(f"kf_{name} must be a finite positive real scalar.")
            try:
                value = float(value)
            except OverflowError as error:
                raise ValueError(f"kf_{name} must be a finite positive real scalar.") from error
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"kf_{name} must be a finite positive real scalar.")
            object.__setattr__(self, name, value)

    @property
    def is_default(self) -> bool:
        """Return whether all existing covariance values are preserved."""
        return (
            self.process_position_scale == 1.0
            and self.process_velocity_scale == 1.0
            and self.measurement_noise_scale == 1.0
            and self.initial_position_scale == 1.0
            and self.initial_velocity_scale == 1.0
            and self.time_unit == "frames"
            and self.reference_dt_s == DEFAULT_REFERENCE_DT_S
        )

    def initial_covariance(self, covariance: np.ndarray, velocity_start: int) -> np.ndarray:
        """Scale P0 by congruence, converting velocities to seconds at creation."""
        if self.initial_position_scale == self.initial_velocity_scale == 1.0 and self.time_unit == "frames":
            return covariance
        factors = np.full(covariance.shape[-1], sqrt(self.initial_position_scale))
        factors[velocity_start:] = sqrt(self.initial_velocity_scale)
        if self.time_unit == "seconds":
            factors[velocity_start:] /= self.reference_dt_s
        return self._congruence(covariance, factors)

    @staticmethod
    def _congruence(covariance: np.ndarray, factors: np.ndarray) -> np.ndarray:
        """Apply diagonal congruence to one covariance or a batch."""
        try:
            with np.errstate(over="raise", invalid="raise"):
                result = covariance * factors[:, None] * factors[None, :]
        except FloatingPointError as error:
            raise ValueError("Kalman noise settings produce an unrepresentable covariance.") from error
        return result

    def _process_factors(self, dim_x: int, velocity_start: int, *, continuous: bool) -> np.ndarray:
        """Return square-root factors for a discrete Q or continuous density."""
        factors = np.full(dim_x, sqrt(self.process_position_scale))
        factors[velocity_start:] = sqrt(self.process_velocity_scale)
        if continuous and self.time_unit == "seconds":
            factors /= sqrt(self.reference_dt_s)
            factors[velocity_start:] /= self.reference_dt_s
        return factors

    def process_covariance(
        self, covariance: np.ndarray, velocity_start: int, *, continuous: bool = False
    ) -> np.ndarray:
        """Scale Q; continuous seconds use L_s = D C Q C D / reference_dt_s.

        D leaves measured states unchanged and divides velocities by the fixed
        reference interval. This PSD density defines a continuous model; its
        integrated covariance need not equal legacy discrete Q at that interval.
        """
        if self.process_position_scale == self.process_velocity_scale and not (
            continuous and self.time_unit == "seconds"
        ):
            if self.process_position_scale == 1.0:
                return covariance
            return covariance * self.process_position_scale
        factors = self._process_factors(covariance.shape[-1], velocity_start, continuous=continuous)
        return self._congruence(covariance, factors)

    def reference_process_diagonal(self, diagonal: np.ndarray, velocity_start: int, *, continuous: bool) -> np.ndarray:
        """Undo process scaling and time conversion after integration is inverted."""
        factors = self._process_factors(diagonal.shape[-1], velocity_start, continuous=continuous)
        return diagonal / factors / factors

    def measurement_covariance(self, covariance: np.ndarray) -> np.ndarray:
        """Scale R in measurement units, which are independent of elapsed time."""
        if self.measurement_noise_scale == 1.0:
            return covariance
        return covariance * self.measurement_noise_scale


def normalize_kalman_options(
    options: Mapping[str, object],
    *,
    variable_dt: bool,
    tracker_name: str | None = None,
    backend: str = "python",
) -> KalmanNoiseConfig:
    """Validate canonical settings, their persisted units, and backend support."""
    if not isinstance(variable_dt, bool):
        raise TypeError("variable_dt must be bool.")
    accepted = {*KALMAN_NOISE_OPTIONS, *KALMAN_TIMING_OPTIONS}
    for option in options:
        if option.startswith("kf_") and option not in accepted:
            raise TypeError(f"Unexpected Kalman option {option!r}.")
    expected_unit = "seconds" if variable_dt else "frames"
    time_unit = options.get("kf_time_unit")
    config = KalmanNoiseConfig(
        **{option.removeprefix("kf_"): options.get(option, 1.0) for option in KALMAN_NOISE_OPTIONS},
        time_unit=expected_unit if time_unit is None else time_unit,
        reference_dt_s=options.get("kf_reference_dt_s", DEFAULT_REFERENCE_DT_S),
    )
    if config.time_unit != expected_unit:
        raise ValueError(
            f"kf_time_unit={config.time_unit!r} conflicts with variable_dt={variable_dt}; expected {expected_unit!r}."
        )
    if variable_dt and backend != "python":
        raise ValueError("The native tracker backend does not support variable_dt=True; use the Python backend.")
    if tracker_name is not None and tracker_name not in KALMAN_TRACKER_NAMES and variable_dt:
        raise ValueError(f"Tracker {tracker_name!r} does not support variable_dt.")
    if not config.is_default and (
        backend != "python" or (tracker_name is not None and tracker_name not in KALMAN_TRACKER_NAMES)
    ):
        raise ValueError("Kalman noise scaling requires a Python Kalman tracker.")
    return config
