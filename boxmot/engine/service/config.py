from __future__ import annotations

import os
from dataclasses import dataclass

CPU_SERVICE_TRACKERS = ("bytetrack", "ocsort", "sfsort")
REID_SERVICE_TRACKERS = (
    "strongsort",
    "botsort",
    "deepocsort",
    "hybridsort",
    "boosttrack",
    "occluboost",
)
SERVICE_TRACKERS_BY_PROFILE = {
    "cpu": CPU_SERVICE_TRACKERS,
    "gpu": REID_SERVICE_TRACKERS,
}
SUPPORTED_SERVICE_PROFILES = tuple(SERVICE_TRACKERS_BY_PROFILE)
SUPPORTED_SERVICE_TRACKERS = CPU_SERVICE_TRACKERS + REID_SERVICE_TRACKERS
ASSOCIATION_FUNCTIONS = ("iou", "giou", "diou", "ciou", "hmiou", "centroid")


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


def _environment_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {raw_value!r}.")


@dataclass(frozen=True, slots=True)
class ServiceSettings:
    """Validated process-wide settings for the tracker service."""

    profile: str = "cpu"
    tracker_type: str = "bytetrack"
    asso_func: str = "iou"
    device: str = "cpu"
    half: bool = False
    reid_weights: str = "osnet_x0_25_msmt17.pt"
    port: int = 8000
    max_streams: int = 256
    stream_ttl_seconds: float = 900.0
    max_detections_per_frame: int = 1_000
    max_classes_per_stream: int = 32
    max_concurrent_updates: int = 4
    max_image_bytes: int = 20_000_000
    max_frame_pixels: int = 33_177_600

    def __post_init__(self) -> None:
        if self.profile not in SUPPORTED_SERVICE_PROFILES:
            available = ", ".join(SUPPORTED_SERVICE_PROFILES)
            raise ValueError(f"Unsupported service profile {self.profile!r}; choose one of: {available}.")
        profile_trackers = SERVICE_TRACKERS_BY_PROFILE[self.profile]
        if self.tracker_type not in profile_trackers:
            available = ", ".join(profile_trackers)
            raise ValueError(
                f"Tracker {self.tracker_type!r} is not available in the {self.profile!r} "
                f"service profile; choose one of: {available}."
            )
        if self.asso_func not in ASSOCIATION_FUNCTIONS:
            available = ", ".join(ASSOCIATION_FUNCTIONS)
            raise ValueError(
                f"Unsupported association function {self.asso_func!r}; choose one of: {available}."
            )
        if not self.device.strip():
            raise ValueError("Service device must not be empty.")
        if not self.reid_weights.strip():
            raise ValueError("Service ReID weights must not be empty.")
        if not 1 <= self.port <= 65_535:
            raise ValueError(f"Service port must be between 1 and 65535, got {self.port}.")
        if not 1 <= self.max_streams <= 100_000:
            raise ValueError(f"max_streams must be between 1 and 100000, got {self.max_streams}.")
        if not 1.0 <= self.stream_ttl_seconds <= 604_800.0:
            raise ValueError(f"stream_ttl_seconds must be between 1 and 604800, got {self.stream_ttl_seconds}.")
        if not 1 <= self.max_detections_per_frame <= 2_000:
            raise ValueError(
                f"max_detections_per_frame must be between 1 and 2000, got {self.max_detections_per_frame}."
            )
        if not 1 <= self.max_classes_per_stream <= 256:
            raise ValueError(f"max_classes_per_stream must be between 1 and 256, got {self.max_classes_per_stream}.")
        if not 1 <= self.max_concurrent_updates <= 64:
            raise ValueError(f"max_concurrent_updates must be between 1 and 64, got {self.max_concurrent_updates}.")
        if not 1_024 <= self.max_image_bytes <= 100_000_000:
            raise ValueError(f"max_image_bytes must be between 1024 and 100000000, got {self.max_image_bytes}.")
        if not 1 <= self.max_frame_pixels <= 1_073_741_824:
            raise ValueError(f"max_frame_pixels must be between 1 and 1073741824, got {self.max_frame_pixels}.")

    @property
    def requires_image(self) -> bool:
        """Whether each request must carry real pixels for ReID and CMC."""

        return self.profile == "gpu"

    @classmethod
    def from_env(cls) -> ServiceSettings:
        """Build settings from ``BOXMOT_SERVICE_*`` environment variables."""

        profile = os.getenv("BOXMOT_SERVICE_PROFILE", "cpu").strip().lower()
        default_tracker = "botsort" if profile == "gpu" else "bytetrack"
        tracker_type = os.getenv("BOXMOT_SERVICE_TRACKER", default_tracker).strip().lower()
        return cls(
            profile=profile,
            tracker_type=tracker_type,
            asso_func=os.getenv("BOXMOT_SERVICE_ASSO_FUNC", "iou").strip().lower(),
            device=os.getenv("BOXMOT_SERVICE_DEVICE", "0" if profile == "gpu" else "cpu").strip(),
            half=_environment_bool("BOXMOT_SERVICE_HALF", profile == "gpu"),
            reid_weights=os.getenv(
                "BOXMOT_SERVICE_REID_WEIGHTS",
                "osnet_x0_25_msmt17.pt",
            ).strip(),
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
                1 if profile == "gpu" else 4,
                minimum=1,
                maximum=64,
            ),
            max_image_bytes=_environment_int(
                "BOXMOT_SERVICE_MAX_IMAGE_BYTES",
                20_000_000,
                minimum=1_024,
                maximum=100_000_000,
            ),
            max_frame_pixels=_environment_int(
                "BOXMOT_SERVICE_MAX_FRAME_PIXELS",
                33_177_600,
                minimum=1,
                maximum=1_073_741_824,
            ),
        )


__all__ = (
    "ASSOCIATION_FUNCTIONS",
    "CPU_SERVICE_TRACKERS",
    "REID_SERVICE_TRACKERS",
    "SERVICE_TRACKERS_BY_PROFILE",
    "SUPPORTED_SERVICE_PROFILES",
    "SUPPORTED_SERVICE_TRACKERS",
    "ServiceSettings",
)
