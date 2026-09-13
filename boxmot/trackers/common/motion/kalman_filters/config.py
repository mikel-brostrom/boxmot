"""Typed configuration for Kalman covariance, timing, and update policies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from math import isfinite
from numbers import Integral, Real
from typing import Any

from boxmot.trackers.common.motion.kalman_filters.noise import (
    KALMAN_NOISE_TRACKER_NAMES,
    KALMAN_TRACKER_NAMES,
    KalmanNoiseConfig,
)

_ADAPTIVE_TRACKERS = frozenset({"boosttrack", "occluboost"})
_REMOVED_OPTIONS = frozenset(
    {
        "kalman_noise",
        "variable_dt",
        "adaptive_kf",
        "is_angular",
        "ams_enabled",
        "ams_alpha0",
        "ams_threshold",
        "ams_buffer_size",
        "ams_shrink_ratio",
    }
)


@dataclass(frozen=True, slots=True)
class AbnormalMotionSuppressionConfig:
    """OccluBoost's AABB Kalman-gain suppression for abnormal observations.

    Args:
        enabled: Apply suppression to abnormal AABB measurement corrections.
        alpha0: Gain multiplier for abnormal motion components, between zero and one.
        threshold: Relative speed-spike threshold above the historical mean.
        buffer_size: Number of observations retained for estimating normal motion.
        shrink_ratio: Box-area ratio below which an observation is treated as occluded.
    """

    enabled: bool = True
    alpha0: float = 0.4
    threshold: float = 0.5
    buffer_size: int = 30
    shrink_ratio: float = 0.75

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("kalman.ams.enabled must be bool.")
        for name in ("alpha0", "threshold", "shrink_ratio"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value < 0:
                raise ValueError(f"kalman.ams.{name} must be a finite non-negative number.")
            if name != "threshold" and value > 1:
                raise ValueError(f"kalman.ams.{name} must be between zero and one.")
            object.__setattr__(self, name, float(value))
        if isinstance(self.buffer_size, bool) or not isinstance(self.buffer_size, Integral) or self.buffer_size < 2:
            raise ValueError("kalman.ams.buffer_size must be an integer of at least two.")
        object.__setattr__(self, "buffer_size", int(self.buffer_size))

    def to_dict(self) -> dict[str, bool | float | int]:
        """Serialize independent scalar settings for YAML and tuning."""
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class KalmanConfig:
    """Kalman settings shared by direct trackers, configuration files, and tuning.

    Optional policies select the owning tracker's defaults when omitted. Filter
    representation and dimensions follow tracker geometry; they are not arbitrary
    interchangeable models. Per-class covariance profiles belong to ``noise``.

    Args:
        noise: Covariance multipliers, unit metadata, and optional class profiles.
        variable_dt: Predict using measured capture intervals in seconds; supported
            by the eight 2D Kalman trackers. False advances one reference frame.
        adaptive_kf: Adapt covariance from innovations in BoostTrack or OccluBoost.
            None selects their default of False; other trackers reject explicit values.
        is_angular: Include object yaw velocity in EagerMOT's 3D state. None selects
            its default of False; other trackers reject explicit values.
        ams: OccluBoost's abnormal-motion gain suppression. None selects its defaults.
            The existing AABB policy is bypassed for OBB tracking.
    """

    noise: KalmanNoiseConfig = field(default_factory=KalmanNoiseConfig)
    variable_dt: bool = False
    adaptive_kf: bool | None = None
    is_angular: bool | None = None
    ams: AbnormalMotionSuppressionConfig | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.noise, KalmanNoiseConfig):
            raise TypeError("kalman.noise must be a KalmanNoiseConfig object.")
        if not isinstance(self.variable_dt, bool):
            raise TypeError("kalman.variable_dt must be bool.")
        for name in ("adaptive_kf", "is_angular"):
            if getattr(self, name) is not None and not isinstance(getattr(self, name), bool):
                raise TypeError(f"kalman.{name} must be bool or None.")
        if self.ams is not None and not isinstance(self.ams, AbnormalMotionSuppressionConfig):
            raise TypeError("kalman.ams must be an AbnormalMotionSuppressionConfig object or None.")

    def resolve(self, tracker_name: str | None = None, *, backend: str = "python") -> KalmanConfig:
        """Resolve tracker defaults and reject policies that its filter cannot use."""
        if tracker_name is not None and tracker_name not in KALMAN_NOISE_TRACKER_NAMES:
            raise ValueError(f"Tracker {tracker_name!r} does not support kalman settings.")
        if self.variable_dt and (backend != "python" or tracker_name not in KALMAN_TRACKER_NAMES | {None}):
            raise ValueError("kalman.variable_dt=True requires a Python 2D Kalman tracker.")
        noise = self.noise.resolve(variable_dt=self.variable_dt)
        if backend != "python" and not noise.is_default:
            raise ValueError("Kalman noise scaling requires a Python Kalman tracker.")
        adaptive, angular, ams = self.adaptive_kf, self.is_angular, self.ams
        if tracker_name is not None:
            if adaptive is not None and tracker_name not in _ADAPTIVE_TRACKERS:
                raise ValueError(f"Tracker {tracker_name!r} does not support kalman.adaptive_kf.")
            if angular is not None and tracker_name != "eagermot":
                raise ValueError(f"Tracker {tracker_name!r} does not support kalman.is_angular.")
            if ams is not None and tracker_name != "occluboost":
                raise ValueError(f"Tracker {tracker_name!r} does not support kalman.ams.")
            if tracker_name in _ADAPTIVE_TRACKERS and adaptive is None:
                adaptive = False
            if tracker_name == "eagermot" and angular is None:
                angular = False
            if tracker_name == "occluboost" and ams is None:
                ams = AbnormalMotionSuppressionConfig()
        if backend != "python" and adaptive:
            raise ValueError("kalman.adaptive_kf=True requires the Python tracker backend.")
        return replace(self, noise=noise, adaptive_kf=adaptive, is_angular=angular, ams=ams)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the grouped runtime configuration without implementation state."""
        result: dict[str, Any] = {"noise": self.noise.to_dict(), "variable_dt": self.variable_dt}
        for name in ("adaptive_kf", "is_angular"):
            if getattr(self, name) is not None:
                result[name] = getattr(self, name)
        if self.ams is not None:
            result["ams"] = self.ams.to_dict()
        return result

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> KalmanConfig:
        """Read authored YAML settings, preserving omitted tracker-specific policies."""
        if not isinstance(values, Mapping):
            raise TypeError("kalman must be a KalmanConfig or a configuration mapping.")
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise TypeError(f"Unexpected kalman field {next(iter(unknown))!r}.")
        payload = dict(values)
        if "noise" in payload and not isinstance(payload["noise"], KalmanNoiseConfig):
            payload["noise"] = KalmanNoiseConfig.from_mapping(payload["noise"])
        if payload.get("ams") is not None and not isinstance(payload["ams"], AbnormalMotionSuppressionConfig):
            if not isinstance(payload["ams"], Mapping):
                raise TypeError("kalman.ams must contain suppression settings.")
            payload["ams"] = AbnormalMotionSuppressionConfig(**payload["ams"])
        return cls(**payload)


def normalize_kalman_config(
    options: Mapping[str, Any], *, tracker_name: str | None = None, backend: str = "python"
) -> KalmanConfig:
    """Resolve scalar engine options or authored groups to the public configuration."""
    from boxmot.trackers.common.config import flatten_tracker_options, nest_tracker_options

    old = set(options) & _REMOVED_OPTIONS
    if old:
        raise TypeError(f"Configure {', '.join(sorted(old))} under 'kalman'.")
    flattened = flatten_tracker_options(options)
    declared = any(name.startswith("kalman.") for name in flattened)
    if tracker_name is not None and tracker_name not in KALMAN_NOISE_TRACKER_NAMES and not declared:
        return KalmanConfig().resolve()
    grouped = nest_tracker_options(flattened).get("kalman", {})
    return KalmanConfig.from_mapping(grouped).resolve(tracker_name, backend=backend)


__all__ = ("AbnormalMotionSuppressionConfig", "KalmanConfig", "normalize_kalman_config")
