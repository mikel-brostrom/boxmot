"""Immutable, instance-local Kalman covariance calibration."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from math import isfinite, sqrt
from numbers import Integral, Real
from typing import Literal

import numpy as np

KALMAN_TRACKER_NAMES = frozenset(
    {"botsort", "boosttrack", "bytetrack", "deepocsort", "hybridsort", "occluboost", "ocsort", "strongsort"}
)
KALMAN_NOISE_TRACKER_NAMES = KALMAN_TRACKER_NAMES | {"eagermot"}
KALMAN_NOISE_OPTIONS = (
    "kalman_noise.process_position_scale",
    "kalman_noise.process_velocity_scale",
    "kalman_noise.measurement_noise_scale",
    "kalman_noise.initial_position_scale",
    "kalman_noise.initial_velocity_scale",
)
KALMAN_TIMING_OPTIONS = ("kalman_noise.time_unit", "kalman_noise.reference_dt_s")
DEFAULT_REFERENCE_DT_S = 1.0 / 30.0


@dataclass(frozen=True, slots=True)
class _ClassNoiseMapping(Mapping[int, "KalmanNoiseConfig"]):
    """Immutable class settings with ordinary pickle support."""

    entries: tuple[tuple[int, KalmanNoiseConfig], ...] = ()

    def __getitem__(self, key: int) -> KalmanNoiseConfig:
        for class_id, value in self.entries:
            if class_id == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[int]:
        return (class_id for class_id, _ in self.entries)

    def __len__(self) -> int:
        return len(self.entries)


def _class_id(value: object, *, serialized: bool = False) -> int:
    """Validate canonical detector IDs, allowing JSON keys when requested."""
    if serialized and isinstance(value, str):
        if not value.isascii() or not value.isdecimal() or str(int(value)) != value:
            raise ValueError("kalman_noise.by_class keys must be canonical non-negative integer class IDs.")
        value = int(value)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not 0 <= value < 2**63:
        raise ValueError("kalman_noise.by_class keys must be non-negative int64 class IDs.")
    return int(value)


@dataclass(frozen=True, slots=True)
class KalmanNoiseConfig:
    """Covariance multipliers and a fixed conversion from frame priors to seconds.

    Position includes every measured state (size, ratio, angle, or confidence),
    while velocity includes their modeled derivatives. The reference interval
    describes the original frame-based priors; it never follows observed dt.

    Args:
        process_position_scale: Multiplier for process covariance in measured states.
        process_velocity_scale: Multiplier for process covariance in derivatives.
        measurement_noise_scale: Multiplier for measurement covariance.
        initial_position_scale: Multiplier for initial measured-state covariance.
        initial_velocity_scale: Multiplier for initial derivative covariance.
        time_unit: Persisted calibration units, ``frames`` or ``seconds``. None
            derives units from the tracker's timing mode when resolved.
        reference_dt_s: Fixed seconds per original reference frame, used when
            converting frame-based priors to seconds.
        by_class: Complete noise settings for individual detector classes.
            Unlisted classes use this object's pooled settings. All classes
            must share its timing units and reference interval.
    """

    process_position_scale: float = 1.0
    process_velocity_scale: float = 1.0
    measurement_noise_scale: float = 1.0
    initial_position_scale: float = 1.0
    initial_velocity_scale: float = 1.0
    time_unit: Literal["frames", "seconds"] | None = None
    reference_dt_s: float = DEFAULT_REFERENCE_DT_S
    by_class: Mapping[int, KalmanNoiseConfig] = field(default_factory=_ClassNoiseMapping)

    def __post_init__(self) -> None:
        if self.time_unit is not None and (
            not isinstance(self.time_unit, str) or self.time_unit not in ("frames", "seconds")
        ):
            raise ValueError("kalman_noise.time_unit must be 'frames', 'seconds', or None.")
        for name in (*[option.removeprefix("kalman_noise.") for option in KALMAN_NOISE_OPTIONS], "reference_dt_s"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise ValueError(f"kalman_noise.{name} must be a finite positive real scalar.")
            try:
                value = float(value)
            except OverflowError as error:
                raise ValueError(f"kalman_noise.{name} must be a finite positive real scalar.") from error
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"kalman_noise.{name} must be a finite positive real scalar.")
            object.__setattr__(self, name, value)
        if not isinstance(self.by_class, Mapping):
            raise TypeError("kalman_noise.by_class must be a mapping of class IDs to KalmanNoiseConfig objects.")
        entries = []
        for key, config in self.by_class.items():
            class_id = _class_id(key)
            if not isinstance(config, KalmanNoiseConfig):
                raise TypeError("kalman_noise.by_class values must be KalmanNoiseConfig objects.")
            if config.by_class:
                raise ValueError("Nested kalman_noise.by_class settings are not supported.")
            if config.reference_dt_s != self.reference_dt_s:
                raise ValueError("Class-specific Kalman noise must share the pooled reference_dt_s.")
            if self.time_unit is not None and config.time_unit is not None and config.time_unit != self.time_unit:
                raise ValueError("Class-specific Kalman noise must share the pooled time_unit.")
            entries.append((class_id, config))
        object.__setattr__(self, "by_class", _ClassNoiseMapping(tuple(sorted(entries))))

    def resolve(self, *, variable_dt: bool = False) -> KalmanNoiseConfig:
        """Resolve fresh units without changing explicitly calibrated units."""
        if not isinstance(variable_dt, bool):
            raise TypeError("variable_dt must be bool.")
        expected = "seconds" if variable_dt else "frames"
        if self.time_unit is not None and self.time_unit != expected:
            raise ValueError(
                f"kalman_noise.time_unit={self.time_unit!r} conflicts with "
                f"variable_dt={variable_dt}; expected {expected!r}."
            )
        children = {class_id: config.resolve(variable_dt=variable_dt) for class_id, config in self.by_class.items()}
        unchanged = self.time_unit == expected and all(children[key] is value for key, value in self.by_class.items())
        return self if unchanged else replace(self, time_unit=expected, by_class=children)

    def for_class(self, class_id: int) -> KalmanNoiseConfig:
        """Return one class's immutable settings, falling back to pooled values."""
        return self.by_class.get(_class_id(class_id), self)

    def to_dict(self) -> dict[str, object]:
        """Return an independent JSON/YAML-compatible configuration mapping."""
        values = {
            option.removeprefix("kalman_noise."): getattr(self, option.removeprefix("kalman_noise."))
            for option in (*KALMAN_NOISE_OPTIONS, *KALMAN_TIMING_OPTIONS)
        }
        if self.by_class:
            values["by_class"] = {str(key): config.to_dict() for key, config in self.by_class.items()}
        return values

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> KalmanNoiseConfig:
        """Parse serialized settings, including string class IDs from YAML/JSON."""
        if not isinstance(values, Mapping):
            raise TypeError("kalman_noise settings must be a mapping.")
        fields = {option.removeprefix("kalman_noise.") for option in (*KALMAN_NOISE_OPTIONS, *KALMAN_TIMING_OPTIONS)}
        unknown = set(values) - fields - {"by_class"}
        if unknown:
            raise TypeError(f"Unexpected kalman_noise field {next(iter(unknown))!r}.")
        payload = dict(values)
        children = payload.pop("by_class", {})
        if not isinstance(children, Mapping):
            raise TypeError("kalman_noise.by_class must be a mapping.")
        parsed = {}
        for key, child in children.items():
            class_id = _class_id(key, serialized=True)
            if class_id in parsed:
                raise ValueError(f"Duplicate kalman_noise.by_class ID {class_id}.")
            parsed[class_id] = child if isinstance(child, cls) else cls.from_mapping(child)
        return cls(**payload, by_class=parsed)

    @property
    def is_default(self) -> bool:
        """Return whether all existing covariance values are preserved."""
        return (
            self.process_position_scale == 1.0
            and self.process_velocity_scale == 1.0
            and self.measurement_noise_scale == 1.0
            and self.initial_position_scale == 1.0
            and self.initial_velocity_scale == 1.0
            and self.time_unit in (None, "frames")
            and self.reference_dt_s == DEFAULT_REFERENCE_DT_S
            and not self.by_class
        )

    def initial_covariance(self, covariance: np.ndarray, velocity_start: int) -> np.ndarray:
        """Scale P0 by congruence, converting velocities to seconds at creation."""
        if self.initial_position_scale == self.initial_velocity_scale == 1.0 and self.time_unit in (None, "frames"):
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
    """Resolve nested or dotted noise settings and validate backend support."""
    if not isinstance(variable_dt, bool):
        raise TypeError("variable_dt must be bool.")
    accepted = {*KALMAN_NOISE_OPTIONS, *KALMAN_TIMING_OPTIONS}
    fields = {option.removeprefix("kalman_noise.") for option in accepted}
    values = {}
    for option in options:
        if not isinstance(option, str):
            raise TypeError("Tracker option names must be strings.")
        if option.startswith("kf_"):
            raise TypeError(f"Unexpected Kalman option {option!r}; use 'kalman_noise' settings.")
        if not option.startswith("kalman_noise."):
            continue
        parts = option.split(".")
        if option in accepted:
            values[parts[1]] = options[option]
        elif len(parts) == 4 and parts[1] == "by_class" and parts[3] in fields:
            class_id = str(_class_id(parts[2], serialized=True))
            child = values.setdefault("by_class", {}).setdefault(class_id, {})
            if parts[3] in child:
                raise ValueError(f"Duplicate Kalman setting {option!r}.")
            child[parts[3]] = options[option]
        else:
            raise TypeError(f"Unexpected Kalman option {option!r}.")
    nested = options.get("kalman_noise")
    if nested is not None and values:
        raise ValueError("Kalman settings cannot be supplied both nested and dotted.")
    if nested is not None and not isinstance(nested, (KalmanNoiseConfig, Mapping)):
        raise TypeError("kalman_noise must be a KalmanNoiseConfig, a mapping, or None.")
    config = (
        nested
        if isinstance(nested, KalmanNoiseConfig)
        else KalmanNoiseConfig.from_mapping(values if nested is None else nested)
    ).resolve(variable_dt=variable_dt)
    if variable_dt and backend != "python":
        raise ValueError("The native tracker backend does not support variable_dt=True; use the Python backend.")
    if tracker_name is not None and tracker_name not in KALMAN_TRACKER_NAMES and variable_dt:
        raise ValueError(f"Tracker {tracker_name!r} does not support variable_dt.")
    declared = "kalman_noise" in options or bool(values)
    if declared and tracker_name is not None and tracker_name not in KALMAN_NOISE_TRACKER_NAMES:
        raise ValueError(f"Tracker {tracker_name!r} does not support kalman_noise settings.")
    if not config.is_default and backend != "python":
        raise ValueError("Kalman noise scaling requires a Python Kalman tracker.")
    return config
