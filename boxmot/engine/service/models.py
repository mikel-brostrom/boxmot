from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BoxType(str, Enum):
    """Geometry selector used by the stable HTTP v1 wire contract."""

    AABB = "aabb"
    OBB = "obb"


class FrameRequest(BaseModel):
    """One ordered frame of canonical detector output."""

    model_config = ConfigDict(extra="forbid")

    frame_id: int = Field(ge=0, le=9_223_372_036_854_775_806)
    width: int | None = Field(
        default=None,
        gt=0,
        le=32_768,
        description="Frame width in pixels; inferred from image_base64, required without an image.",
    )
    height: int | None = Field(
        default=None,
        gt=0,
        le=32_768,
        description="Frame height in pixels; inferred from image_base64, required without an image.",
    )
    frame_rate: int = Field(default=30, ge=1, le=240)
    box_type: BoxType = BoxType.AABB
    # Keep integer JSON numbers intact for IDs while continuing to accept the
    # established numeric row wire format for geometry and scores.
    detections: list[list[float | int]] = Field(default_factory=list, max_length=2_000)
    image_base64: str | None = Field(
        default=None,
        description="Base64-encoded JPEG or PNG frame; required by the GPU/ReID service profile.",
    )

    @model_validator(mode="after")
    def validate_frame_dimensions(self) -> FrameRequest:
        """Require explicit dimensions only when no image can supply them."""
        if self.image_base64 is None and (self.width is None or self.height is None):
            raise ValueError("width and height are required when image_base64 is not provided.")
        return self


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


__all__ = ("BoxType", "FrameRequest", "FrameResponse", "ReadinessResponse")
