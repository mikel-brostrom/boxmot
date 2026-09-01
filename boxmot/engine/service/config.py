from __future__ import annotations

import os
from dataclasses import dataclass

SUPPORTED_SERVICE_TRACKERS = ("bytetrack", "ocsort")


def _environment_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw_value = os.getenv(name)
    try:
        value = default if raw_value is None else int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}.") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}, got {value}.")
    return value


def _environment_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw_value = os.getenv(name)
    try:
        value = default if raw_value is None else float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric, got {raw_value!r}.") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}, got {value}.")
    return value


@dataclass(frozen=True, slots=True)
class ServiceSettings:
    """Validated process-wide settings for the tracker service."""

    tracker_type: str = "bytetrack"
    port: int = 8000
    max_streams: int = 256
    stream_ttl_seconds: float = 900.0
    max_detections_per_frame: int = 1_000
    max_classes_per_stream: int = 32
    max_concurrent_updates: int = 4

    def __post_init__(self) -> None:
        if self.tracker_type not in SUPPORTED_SERVICE_TRACKERS:
            available = ", ".join(SUPPORTED_SERVICE_TRACKERS)
            raise ValueError(
                f"Unsupported service tracker {self.tracker_type!r}; choose one of: {available}."
            )
        if not 1 <= self.port <= 65_535:
            raise ValueError(f"Service port must be between 1 and 65535, got {self.port}.")
        if not 1 <= self.max_streams <= 100_000:
            raise ValueError(f"max_streams must be between 1 and 100000, got {self.max_streams}.")
        if not 1.0 <= self.stream_ttl_seconds <= 604_800.0:
            raise ValueError(
                "stream_ttl_seconds must be between 1 and 604800, "
                f"got {self.stream_ttl_seconds}."
            )
        if not 1 <= self.max_detections_per_frame <= 2_000:
            raise ValueError(
                "max_detections_per_frame must be between 1 and 2000, "
                f"got {self.max_detections_per_frame}."
            )
        if not 1 <= self.max_classes_per_stream <= 256:
            raise ValueError(
                "max_classes_per_stream must be between 1 and 256, "
                f"got {self.max_classes_per_stream}."
            )
        if not 1 <= self.max_concurrent_updates <= 64:
            raise ValueError(
                "max_concurrent_updates must be between 1 and 64, "
                f"got {self.max_concurrent_updates}."
            )

    @classmethod
    def from_env(cls) -> ServiceSettings:
        """Build settings from ``BOXMOT_SERVICE_*`` environment variables."""

        tracker_type = os.getenv("BOXMOT_SERVICE_TRACKER", "bytetrack").strip().lower()
        return cls(
            tracker_type=tracker_type,
            port=_environment_int("BOXMOT_SERVICE_PORT", 8000, minimum=1, maximum=65_535),
            max_streams=_environment_int(
                "BOXMOT_SERVICE_MAX_STREAMS",
                256,
                minimum=1,
                maximum=100_000,
            ),
            stream_ttl_seconds=_environment_float(
                "BOXMOT_SERVICE_STREAM_TTL_SECONDS",
                900.0,
                minimum=1.0,
                maximum=604_800.0,
            ),
            max_detections_per_frame=_environment_int(
                "BOXMOT_SERVICE_MAX_DETECTIONS",
                1_000,
                minimum=1,
                maximum=2_000,
            ),
            max_classes_per_stream=_environment_int(
                "BOXMOT_SERVICE_MAX_CLASSES",
                32,
                minimum=1,
                maximum=256,
            ),
            max_concurrent_updates=_environment_int(
                "BOXMOT_SERVICE_MAX_CONCURRENT_UPDATES",
                4,
                minimum=1,
                maximum=64,
            ),
        )


__all__ = ("SUPPORTED_SERVICE_TRACKERS", "ServiceSettings")
