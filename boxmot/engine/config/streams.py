"""Validated authored settings for one continuous network camera."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from boxmot.engine.config.runtime import build_mode_namespace
from boxmot.utils.config import ConfigurationError, load_yaml_mapping


@dataclass(frozen=True, slots=True)
class StreamConfig:
    """Camera identity and deployment policy; credentials stay in the environment."""

    camera_id: str
    source_url_env: str = "BOXMOT_CAMERA_URL"
    attributes: Any = "person-appearance-vllm-cpu"
    attributes_host: str | None = None
    tracking: dict[str, Any] = field(default_factory=dict)
    output_dir: Path = Path("runs/streams")
    health_port: int = 8080
    capture_timeout_s: float = 10
    timestamp_source: str = "capture"
    reconnect_max_s: float = 30
    ready_timeout_s: float = 30
    rotation_bytes: int = 64 * 1024 * 1024
    rotation_seconds: float = 3600
    retention_days: int = 7
    max_pending_jobs: int = 1000
    max_crop_bytes: int = 256 * 1024 * 1024
    max_frame_bytes: int = 32 * 1024 * 1024

    def __post_init__(self) -> None:
        if not isinstance(self.camera_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,62}", self.camera_id):
            raise ConfigurationError("camera_id must contain 1–63 letters, digits, underscores or hyphens.")
        if not isinstance(self.source_url_env, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.source_url_env):
            raise ConfigurationError("source_url_env must be an environment variable name.")
        if self.timestamp_source not in ("capture", "media"):
            raise ConfigurationError("timestamp_source must be 'capture' or 'media'.")
        for name in (
            "health_port",
            "rotation_bytes",
            "retention_days",
            "max_pending_jobs",
            "max_crop_bytes",
            "max_frame_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConfigurationError(f"{name} must be a positive integer.")
        if self.health_port > 65535:
            raise ConfigurationError("health_port must be <= 65535.")
        for name in ("capture_timeout_s", "reconnect_max_s", "ready_timeout_s", "rotation_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ConfigurationError(f"{name} must be finite and positive.")
        if not isinstance(self.tracking, dict):
            raise ConfigurationError("tracking must be a mapping of tracking settings.")
        allowed = {
            "detector",
            "tracker",
            "reid",
            "device",
            "half",
            "conf",
            "iou",
            "imgsz",
            "classes",
            "geometry",
            "tracker_config",
            "tracker_backend",
            "per_class",
            "variable_dt",
        }
        unknown = set(self.tracking) - allowed
        if unknown:
            raise ConfigurationError(f"Unsupported streaming tracking options: {', '.join(sorted(unknown))}.")
        if self.tracking.get("tracker") == "edgetam":
            raise ConfigurationError("Streaming requires a detector-driven tracker.")
        object.__setattr__(self, "output_dir", Path(self.output_dir))

    def tracking_args(self) -> Any:
        """Use canonical track normalization without CLI preview defaults."""
        return build_mode_namespace(
            "track",
            {
                "detector": "yolo26n",
                "tracker": "bytetrack",
                "device": "cpu",
                **self.tracking,
                "show": False,
                "save": False,
                "save_json": False,
                "save_txt": False,
            },
        )


def load_stream_config(path: str | Path) -> StreamConfig:
    """Load one strict YAML configuration with no embedded camera credentials."""
    values = load_yaml_mapping(Path(path))
    unknown = set(values) - {item.name for item in fields(StreamConfig)}
    if unknown:
        raise ConfigurationError(f"Unknown stream configuration fields: {', '.join(sorted(unknown))}.")
    if "camera_id" not in values:
        raise ConfigurationError("Stream configuration requires camera_id.")
    return StreamConfig(**values)
