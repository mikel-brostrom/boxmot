from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from boxmot.core.box_schema import BoxType


class FrameRequest(BaseModel):
    """One ordered frame of canonical detector output."""

    model_config = ConfigDict(extra="forbid")

    frame_id: int = Field(ge=0, le=9_223_372_036_854_775_806)
    width: int = Field(gt=0, le=32_768)
    height: int = Field(gt=0, le=32_768)
    frame_rate: int = Field(default=30, ge=1, le=240)
    box_type: BoxType = BoxType.AABB
    detections: list[list[float]] = Field(default_factory=list, max_length=2_000)
    image_base64: str | None = Field(
        default=None,
        description="Base64-encoded JPEG or PNG frame; required by the GPU/ReID service profile.",
    )


class FrameResponse(BaseModel):
    """Tracks produced for one frame."""

    frame_id: int
    next_frame_id: int
    box_type: BoxType
    track_columns: list[str]
    tracks: list[list[float | int]]
    replayed: bool = False


class ReadinessResponse(BaseModel):
    """Current service capacity and tracker configuration."""

    status: str
    profile: str
    tracker: str
    device: str
    requires_image: bool
    active_streams: int
    max_streams: int


__all__ = ("FrameRequest", "FrameResponse", "ReadinessResponse")
