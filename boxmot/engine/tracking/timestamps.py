"""Consistent capture timestamps at OpenCV-backed engine input boundaries."""

from math import isfinite


class LiveTimestamps:
    """Resolve one live connection without trusting an advertised frame rate.

    ``capture`` uses elapsed monotonic time at decoder receipt, independently of
    detector speed and wall-clock corrections. ``media`` requires trustworthy,
    strictly increasing PTS; invalid PTS ends the connection instead of silently
    switching clocks. Create a new resolver with each tracking session.
    """

    def __init__(self, source: str = "capture") -> None:
        if source not in {"capture", "media"}:
            raise ValueError("Live timestamp source must be 'capture' or 'media'.")
        self.source = source
        self._origin_s: float | None = None
        self._last_s: float | None = None

    def resolve(self, *, captured_at: float, position_ms: float | None = None) -> float:
        """Return connection-relative seconds, preserving skipped-frame gaps."""
        value = captured_at if self.source == "capture" else None if position_ms is None else position_ms / 1000
        if value is None or not isfinite(value) or value < 0:
            raise ValueError(f"Live {self.source} timestamps must be finite and nonnegative.")
        if self._last_s is not None and value <= self._last_s:
            raise ValueError(f"Live {self.source} timestamps must increase strictly.")
        if self._origin_s is None:
            self._origin_s = value
        self._last_s = value
        return value - self._origin_s


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
