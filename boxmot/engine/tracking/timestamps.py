"""Consistent capture timestamps at OpenCV-backed engine input boundaries."""

from math import isfinite


class SourceTimestamps:
    """Prefer media PTS and use nominal FPS when those timestamps are unusable.

    Timing availability is latched on the first frame. A valid FPS or positive
    PTS enables timestamps; zero/invalid PTS with unknown FPS stays untimed.
    Once enabled, timestamps must increase. Nominal FPS advances across skipped
    source frames and reconnects; processing wall-clock time is never used.
    """

    def __init__(self, fps: float) -> None:
        self._period = 1.0 / fps if isfinite(fps) and fps > 0.0 else None
        if self._period is not None and not isfinite(self._period):
            self._period = None
        self._enabled: bool | None = None
        self._last_timestamp_s: float | None = None
        self._last_frame_index = 0

    def resolve(self, position_ms: float, *, frame_index: int) -> float | None:
        """Resolve PTS using the original decoded index, including stride gaps."""
        media_timestamp = position_ms / 1000.0 if isfinite(position_ms) and position_ms >= 0.0 else None
        if self._enabled is None:
            self._enabled = self._period is not None or (media_timestamp is not None and media_timestamp > 0.0)
        if not self._enabled:
            return None
        if media_timestamp is not None and (self._last_timestamp_s is None or media_timestamp > self._last_timestamp_s):
            timestamp_s = media_timestamp
        elif self._period is not None:
            timestamp_s = (
                frame_index * self._period
                if self._last_timestamp_s is None
                else self._last_timestamp_s + (frame_index - self._last_frame_index) * self._period
            )
        else:
            raise ValueError("Video source stopped providing increasing timestamps and has no nominal FPS fallback.")
        if not isfinite(timestamp_s) or (self._last_timestamp_s is not None and timestamp_s <= self._last_timestamp_s):
            raise ValueError("Video source cannot provide a finite, increasing timestamp.")
        self._last_timestamp_s = timestamp_s
        self._last_frame_index = frame_index
        return timestamp_s
